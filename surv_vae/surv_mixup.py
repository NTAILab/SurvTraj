import torch
from .vae import VAE
from .beran import BENK
from .surv_weights import PiHead
from . import DEVICE
from typing import Dict, Tuple, Optional
import time
import numpy as np
from numba import njit, bool_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .smoother import ConvStepSmoother, SoftminStepSmoother

# makes background for the Beran estimator
# the one is partially censored


@njit
def dataset_generator(X, T, D, background_size, batch_n):
    # idx = np.arange(X.shape[0])
    # np.random.shuffle(idx)
    # X, T, D = X[idx], T[idx], D[idx]
    dim = X.shape[1]
    N = background_size
    n = X.shape[0] - N
    x_bg = np.empty((batch_n, N, dim), dtype=np.float64)
    t_bg = np.empty((batch_n, N), dtype=np.float64)
    d_bg = np.empty((batch_n, N), dtype=np.int64)
    x_target = np.empty((batch_n, n, dim), dtype=np.float64)
    d_target = np.empty((batch_n, n), dtype=np.int64)
    t_target = np.empty((batch_n, n), dtype=np.float64)
    idx = np.arange(X.shape[0])
    for i in range(batch_n):
        back_idx = np.random.choice(idx, N, False)
        target_mask = np.ones(X.shape[0]).astype(bool_)
        target_mask[back_idx] = 0
        target_idx = np.argwhere(target_mask)[:, 0]
        x_bg[i, ...] = X[back_idx]
        t_bg[i, :] = T[back_idx]
        d_bg[i, :] = D[back_idx]
        x_target[i, ...] = X[target_idx]
        t_target[i, :] = T[target_idx]
        d_target[i, :] = D[target_idx]
    return x_bg, t_bg, d_bg, x_target, t_target, d_target

# last element is tha fake zero, so number of the elements in the
# output is the same as in the input
# last step is zeroed at all as it does not take part in the integral
@njit
def calc_t_diff(t_labels):
    result = np.empty(t_labels.shape[0] - 1, dtype=np.float64)
    # zero_step_mask = np.zeros(t_labels.shape[0] - 1, dtype=bool)
    # result[-1] = 0
    # zero_step_mask[-1] = True
    last_diff = t_labels[-2] - t_labels[-1]
    for i in range(1, t_labels.shape[0]):
        cur_diff = t_labels[-i] - t_labels[-i - 1]
        if cur_diff > 1e-8:
            result[-i] = cur_diff
            last_diff = cur_diff
        else:
            result[-i] = last_diff
    return result#, zero_step_mask

def c_ind_loss(T_true, D, T_calc, sigma_temp):
    assert T_true.ndim == D.ndim == T_calc.ndim == 1
    T_diff = T_true[:, None] - T_true[None, :]
    T_mask = T_diff < 0
    sigma = torch.nn.functional.sigmoid(sigma_temp * (T_calc[None, :] - T_calc[:, None]))
    C = torch.sum(T_mask * sigma * D[:, None]) / torch.sum(T_mask * D[:, None])
    return C


class SurvivalMixup(torch.nn.Module):

    def __init__(self, vae_kw: Dict,
                 samples_num: int,
                 batch_num: int = 64,
                 epochs: int = 100,
                 lr_rate: float = 0.001,
                 benk_vae_loss_rat: float = 0.2,
                 c_ind_temp: float = 1.0,
                 gumbel_tau: float = 1.0,
                 train_bg_part: float = 0.6,
                 cens_cls_model = None,
                 batch_load: Optional[int] = None) -> None:
        super().__init__()
        self.vae = VAE(**vae_kw)
        self.samples_num = samples_num
        self.pi_head = None
        self.benk = BENK(self.vae.latent_dim)
        self.batch_num = batch_num
        self.epochs = epochs

        self.lr_rate = lr_rate
        # self.vae_loss = self.vae.loss_fn
        self.vae_loss = self.vae.mmd_loss
        # self.benk_loss = torch.nn.MSELoss()
        self.benk_loss = c_ind_loss
        self.c_ind_temp = c_ind_temp
        self.benk_ratio = benk_vae_loss_rat
        self.batch_load = batch_load
        
        # self.smoother = ConvStepSmoother()
        self.smoother = SoftminStepSmoother()
        if cens_cls_model is not None:
            assert hasattr(cens_cls_model, 'predict_proba')
            assert hasattr(cens_cls_model, 'fit')
        self.cens_model = cens_cls_model
        self.uncens_part = None
        self.train_bg_part = train_bg_part
        self.gumbel_tau = gumbel_tau

    def _lazy_init(self, x: np.ndarray, y: np.recarray):
        self.pi_head = PiHead(None)
        self.vae._lazy_init(x)

    def _set_background(self, X: torch.Tensor, T: torch.Tensor, D: torch.Tensor):
        self.T_bg, sort_args = torch.sort(T)
        self.X_bg = X[sort_args]
        self.D_bg = D[sort_args]
        # T_diff is used for the integral computation, SO:
        # the same time labels will have the same diff metric:
        # (1, 2, 2, 3) -> (1, 1, 1, 0)
        t_diff = calc_t_diff(self.T_bg.cpu().numpy())
        assert np.any(t_diff > 0)
        self.T_diff_int = self.T_bg[1:] - self.T_bg[:-1]
        self.T_diff_prob = torch.tensor(t_diff, dtype=torch.get_default_dtype(), device=DEVICE)
        self.zero_mask = self.T_diff_prob < 1e-8
        

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), self.lr_rate)
    
    def forward_exp_time(self, x: torch.Tensor) -> torch.Tensor:
        x_latent = self.vae.encoder(x)#self.vae.mu_nn(self.vae.encoder(x))
        surv_func, _ = self.benk(*self._prepare_bg_data(x.shape[0]), x_latent)
        E_T = self.calc_exp_time(surv_func)
        return E_T
        
    def fit(self, x: np.ndarray, y: np.recarray,
            log_dir:Optional[str]=None, f_calc_val:bool = False) -> 'SurvivalMixup':
        assert x.shape[0] == y.shape[0]
        self._lazy_init(x, y)
        optimizer = self._get_optimizer()
        start_time = time.time()
        self.train()
        
        if log_dir is not None:
            summary_writer = SummaryWriter(log_dir)

        # delta_mask = y['censor'] == 1
        # x_train, t_train = x[delta_mask], y['time'][delta_mask]
        back_size = int(self.train_bg_part * x.shape[0])
        print('Task volume:', x.shape[0] - back_size)
        print('Background volume:', back_size)
        sub_batch_len = self.batch_load if self.batch_load is not None else x.shape[0] - back_size
        
        for e in range(1, self.epochs + 1):
            cum_loss, cum_x_loss, cum_reg_loss, cum_benk_loss, cum_val_t_loss = 0, 0, 0, 0, 0
            i = 0
            train_T = y['time'].copy()
            train_D = y['censor'].copy()
            data = dataset_generator(x, train_T, train_D, back_size, self.batch_num)
            X_back = torch.from_numpy(data[0]).type(
                torch.get_default_dtype()).to(DEVICE)
            T_back = torch.from_numpy(data[1]).type(
                torch.get_default_dtype()).to(DEVICE)
            D_back = torch.from_numpy(data[2]).type(torch.int64).to(DEVICE)
            X_target = torch.from_numpy(data[3]).type(
                torch.get_default_dtype())
            T_target = torch.from_numpy(data[4]).type(
                torch.get_default_dtype())
            D_target = torch.from_numpy(data[5]).type(torch.int64)
            dataset = TensorDataset(X_back, T_back, D_back, X_target, T_target, D_target)
            data_loader = DataLoader(dataset, 1, shuffle=False)

            prog_bar = tqdm(data_loader,
                            f'Epoch {e}', unit='task', ascii=True)

            for x_b, t_b, d_b, x_t, t_t, d_t in prog_bar:
                x_b.squeeze_(0)
                t_b.squeeze_(0)
                d_b.squeeze_(0)
                x_t.squeeze_(0)
                t_t.squeeze_(0)
                d_t.squeeze_(0)
                
                optimizer.zero_grad()
                self._set_background(x_b, t_b, d_b)
                
                target_ds = TensorDataset(x_t, t_t, d_t)
                target_loader = DataLoader(target_ds, sub_batch_len, False)
                benk_loss, vae_loss, x_loss, regularizer = 0, 0, 0, 0
                
                for x_t_b, t_t_b, d_t_b in target_loader:
                    x_t_b, t_t_b, d_t_b = x_t_b.to(DEVICE), t_t_b.to(DEVICE), d_t_b.to(DEVICE)
                
                    x_est, E_T, z, mu, sigma = self(x_t_b)

                    # uncens_mask = d_t == 1
                    # benk_loss = self.benk_loss(E_T[uncens_mask], t_t[uncens_mask])
                    benk_loss += self.benk_loss(t_t_b, d_t_b, E_T, self.c_ind_temp)
                    # vae_loss, x_loss, regularizer = self.vae_loss(x_t, x_est, mu, sigma)
                
                    # z = z.flatten(end_dim=-2)
                    cur_vae_loss, cur_x_loss, cur_regularizer = self.vae_loss(x_t_b, z, x_est)
                    vae_loss += cur_vae_loss
                    x_loss += cur_x_loss
                    regularizer += cur_regularizer
                
                tl_len = len(target_loader)
                benk_loss /= tl_len
                vae_loss /= tl_len
                x_loss /= tl_len
                regularizer /= tl_len
                
                loss = self.benk_ratio * (-benk_loss) + (1 - self.benk_ratio) * vae_loss

                loss.backward()
                
                # gradients check
                # for p in self.parameters():
                #     # print('grad max:', torch.max(torch.abs(p.grad)))
                #     grad_mask = torch.logical_not(torch.isfinite(p.grad))
                #     if torch.any(grad_mask).item():
                #         print('Bad values encountered in the gradients!')
                #     p.grad[grad_mask] = 0

                optimizer.step()

                cum_loss += loss.item()
                cum_x_loss += x_loss
                cum_reg_loss += regularizer
                cum_benk_loss += benk_loss.item()
                i += 1
                epoch_metrics = {
                    'Loss': cum_loss / i,
                    'Recon': cum_x_loss / i,
                    'Regul': cum_reg_loss / i,
                    'C ind': cum_benk_loss / i
                }
                if f_calc_val:
                    raise NotImplementedError()
                    # difference between the x time estim and
                    # estim times of the generated points
                    with torch.no_grad():
                        E_T_val = self.forward_exp_time(x_est)
                        cum_val_t_loss += torch.mean((E_T_val - t_t) ** 2).item()
                        epoch_metrics['Val T mse'] = cum_val_t_loss / i
                prog_bar.set_postfix(epoch_metrics)
            if log_dir is not None:
                summary_writer.add_scalars('train_metrics', epoch_metrics, e)
        self._set_background(
            torch.from_numpy(x).type(
                torch.get_default_dtype()).to(DEVICE),
            torch.from_numpy(y['time'].copy()).type(
                torch.get_default_dtype()).to(DEVICE),
            torch.from_numpy(y['censor'].copy()).type(torch.int).to(DEVICE)
        )
        time_elapsed = time.time() - start_time
        print('Training time:', round(time_elapsed, 1), 's.')
        if log_dir is not None:
            summary_writer.close()
        self.eval()
        if self.cens_model is not None:
            self.fit_cens_model(x, train_T, train_D)
        else:
            self.uncens_part = np.sum(train_D) / train_D.shape[0]
        return self

    def fit_cens_model(self, x: np.ndarray, t: np.ndarray, d: np.ndarray):
        train_data = np.concatenate((x, t[:, None]), axis=-1)
        self.cens_model.fit(train_data, d)
        
    def _prepare_bg_data(self, batch_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code_bg = self.vae.encoder(self.X_bg)
        mu_bg = code_bg#self.vae.mu_nn(code_bg)
        X_bg = mu_bg[None, ...].expand((batch_num, -1, -1))
        # T_bg = self.T_bg[None, :].expand((x.shape[0], -1))
        D_bg = self.D_bg[None, :].expand((batch_num, -1))
        return X_bg, D_bg

    def calc_exp_time(self, surv_func: torch.Tensor) -> torch.Tensor:
        integral = self.T_bg[None, 0] + \
            torch.sum(surv_func[:, :-1] * self.T_diff_int[None, :], dim=-1)
        return integral # (batch)
    
    def gen_exp_time(self, surv_steps: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(surv_steps, 1e-10)
        softmax = torch.nn.functional.gumbel_softmax(logits, self.gumbel_tau, dim=-1, hard=True)
        # time_arg = torch.argmax(softmax, dim=-1)
        # times = torch.take_along_dim(self.T_bg, time_arg)
        times = torch.sum(self.T_bg[None, :-1] * softmax, -1)
        return times
    
    def calc_pdf(self, x: torch.Tensor, mu: torch.Tensor, sigma_log: torch.Tensor) -> torch.Tensor:
        x_cnt = x - mu
        sigma = torch.exp(sigma_log / 2) # (x_n, m)
        det_sqrt = torch.prod(sigma, dim=-1) # (x_n)
        cov_mtx_inv = 1 / torch.exp(sigma_log) # (x_n, m)
        return 1 / (np.power(2 * np.pi, x.shape[-1] / 2) * det_sqrt) *\
            torch.exp(-1 / 2 * torch.sum(x_cnt * cov_mtx_inv * x_cnt, dim=-1))
            
    def surv_steps_to_proba(self, surv_steps: torch.Tensor) -> torch.Tensor:
        t_diff_corr = self.T_diff_prob.clone()
        mask = self.zero_mask.clone()
        t_diff_corr[mask] = 1
        t_diff_corr = t_diff_corr[None, :]
        mask = mask[None, :]
        if surv_steps.ndim == 3:
            t_diff_corr = t_diff_corr[None, ...]
            mask = mask[None, ...]
        proba = surv_steps / t_diff_corr
        mask = mask.broadcast_to(proba.shape)
        proba[mask] = 0
        return proba
        
    def get_surv_proba(self, surv_steps: torch.Tensor, time: torch.Tensor, single_time: bool):
        step_idx = torch.searchsorted(self.T_bg, time) # (t_n)
        idx_pattern = torch.arange(0, self.surv_prob_mask.shape[0], device=DEVICE) - self.surv_prob_mask.shape[0] // 2
        idx_pattern = idx_pattern[None, :] + step_idx[:, None] # (x_n, mask_len)
        clamp_mask = torch.logical_or(idx_pattern < 0, idx_pattern >= self.T_bg.shape[0])
        idx_pattern[clamp_mask] = 0
        if single_time:
            idxes = idx_pattern[:, None, :]
            steps_shape = (surv_steps.shape[0], surv_steps.shape[1], self.surv_prob_mask.shape[0])
        else:
            idxes = idx_pattern.ravel()[None, None, :]
            steps_shape = (surv_steps.shape[0], surv_steps.shape[1], time.shape[0], self.surv_prob_mask.shape[0])
        step = torch.take_along_dim(surv_steps, dim=-1, indices=idxes) # (x_n, z_n, t_n * mask_len)
        if not single_time:
            step = step.reshape(steps_shape)
        prob_mask = self.surv_prob_mask[None, :].broadcast_to((time.shape[0], self.surv_prob_mask.shape[0])).clone()
        prob_mask[clamp_mask] = 0
        prob_mask = prob_mask / torch.sum(prob_mask, dim=-1, keepdim=True)
        if single_time:
            prob_mask = prob_mask[:, None, :]
        else:
            prob_mask = prob_mask[None, None, ...]
        proba = torch.sum(step * prob_mask, dim=-1)
        return proba

    # from x to its reconstrucrion and estimated time
    def forward(self, x: torch.Tensor):
        code = self.vae.encoder(x)
        z_smp, mu, sigma = self.vae.latent_space_from_code(
            code, self.samples_num)  # (x_n, z_n, m)
        
        mu_out, sigma_out = mu, sigma
        
        # z_smp_flt = z_smp.flatten(end_dim=-2) # (x_n * z_n, m)
        surv_func, surv_steps = self.benk(*self._prepare_bg_data(x.shape[0]), mu) # (batch, t_n)
        E_T = self.gen_exp_time(surv_steps) # (x_n)
        
        _, surv_steps = self.benk(*self._prepare_bg_data(x.shape[0] * self.samples_num), z_smp.flatten(end_dim=-2)) # (x_n * z_n, t_n)
        # surv_proba = self.surv_steps_to_proba(surv_steps)
        # surv_proba = surv_proba.reshape(x.shape[0], self.samples_num, -1) # (x_n, z_n, t_n)
        surv_steps = surv_steps.reshape(x.shape[0], self.samples_num, -1) # (x_n, z_n, t_n)
        # surv_steps = surv_steps.reshape(x.shape[0], self.samples_num, -1) # (x_n, z_n, t_n)
        
        # E_T = self.calc_exp_time(surv_func)  # (x_n)
        
        mu = mu[:, None, :].expand(-1, z_smp.shape[1], -1)
        sigma = sigma[:, None, :].expand(-1, z_smp.shape[1], -1)
        pdf = self.calc_pdf(z_smp.flatten(end_dim=-2), mu.flatten(end_dim=-2), sigma.flatten(end_dim=-2)).reshape(x.shape[0], self.samples_num)  # (x_n, z_n)
        
        prob_time = self.T_bg[:-1] + self.T_diff_prob / 2
        surv_proba = self.smoother(surv_steps, prob_time, E_T, True)
        
        
        # gumbel_weights = F.gumbel_softmax(torch.log(surv_steps), tau=1)
    
        
        pi = self.pi_head(surv_proba, pdf, E_T)
        
        # pi = self.pi_head(step, pdf)  # (batch, z_n, 1)
        z_est = torch.sum(pi[..., None] * z_smp, dim=1)  # (x_n, m)

        x_est = self.vae.decoder(z_est)

        return x_est, E_T, z_smp[:, 0, ...], mu_out, sigma_out
    
    def forward_trajectory(self, x: torch.Tensor, points_num: int, t_min: float, t_max: float, f_single_samples_set=True):
        sampler_mlp = 1 if f_single_samples_set else points_num
        # self.samples_num *= 2
        code = self.vae.encoder(x)
        # t_n is the background size
        
        z_smp, mu, sigma = self.vae.latent_space_from_code(
            code, self.samples_num * sampler_mlp) # (x_n, z_n, m)
        
        # z_smp_flt = z_smp.flatten(end_dim=-2) # (x_n * z_n * p_n, m)
        _, surv_steps = self.benk(*self._prepare_bg_data(x.shape[0] * self.samples_num * sampler_mlp), z_smp.flatten(end_dim=-2)) # (x_n * z_n, t_n)
        surv_proba = surv_steps
        surv_proba = surv_proba.reshape(x.shape[0], self.samples_num * sampler_mlp, -1) # (x_n, z_n, t_n)
        # surv_steps = surv_steps.reshape(x.shape[0], self.samples_num, -1) # (x_n, z_n, t_n)
        
        # threshold = torch.tensor([[surv_threshold]], requires_grad=True, dtype=torch.get_default_dtype())
        # last_observed = surv_func.shape[1] - 1 - torch.searchsorted(surv_func.flip(-1), threshold.repeat(z_smp_flt.shape[0], 1)).clip_(max=surv_func.shape[1] - 1)
        # # last_observed = torch.amax(torch.argwhere(surv_func > surv_threshold), -1, keepdim=True)
        # T_last = self.T_bg[None, :].expand(z_smp_flt.shape[0], -1).gather(1, last_observed) # (x_n * z_n * p_n, 1)
        
        T = torch.tensor([((t_max - t_min) / (points_num - 1))], device=DEVICE).repeat(points_num)
        T[0] = t_min
        T = torch.cumsum(T, dim=-1) # (p_n)
        # T = torch.tensor([7.5], device=DEVICE)
        
        prob_time = self.T_bg[:-1] + self.T_diff_prob / 2
        if not f_single_samples_set:
            surv_proba = surv_proba.reshape(x.shape[0] * self.samples_num, sampler_mlp, -1)
            surv_proba.swapaxes_(0, 1)
        surv_proba = self.smoother(surv_proba, prob_time, T, not f_single_samples_set)
        if not f_single_samples_set:
            surv_proba.swapaxes_(0, 1)
            surv_proba = surv_proba.reshape(x.shape[0], self.samples_num, sampler_mlp)
        
        mu = mu[:, None, :].expand(-1, z_smp.shape[1], -1)
        sigma = sigma[:, None, :].expand(-1, z_smp.shape[1], -1)
        pdf = self.calc_pdf(z_smp.flatten(end_dim=-2), mu.flatten(end_dim=-2), sigma.flatten(end_dim=-2)).reshape(x.shape[0], self.samples_num * sampler_mlp, 1)
        if f_single_samples_set:
            pdf = pdf.expand(-1, -1, points_num)    
        else:
            pdf = pdf.reshape(x.shape[0], self.samples_num, sampler_mlp)
        
        pi = self.pi_head(surv_proba, pdf, T) # (x_n, z_n, p_n)
       
        if f_single_samples_set:
            z_smp = z_smp[..., None, :].expand(x.shape[0], self.samples_num, points_num, -1) # (x_n, z_n, p_n, m)
        else:
            z_smp = z_smp.reshape(x.shape[0], self.samples_num, sampler_mlp, -1)
        z_est = torch.sum(pi[..., None] * z_smp, dim=1)  # (x_n, p_n, m)
        
        x_est = self.vae.decoder(z_est)
        return x_est, T[None, :].repeat(x.shape[0], 1)
        
    def interpolate_code(self, x_1: torch.Tensor, x_2: torch.Tensor, num: int):
        code1 = self.vae.encoder(x_1) # (batch, m)
        code2 = self.vae.encoder(x_2) # (batch, m)
        l_space = torch.linspace(0, 1, num, device=DEVICE)[None, :, None] # (1, num, 1)
        code_inp = (1 - l_space) * code1[:, None, :] + l_space * code2[:, None, :] # (batch, num, m)
        
        out_shape = code_inp.shape
        code_inp = code_inp.flatten(end_dim=-2) # (batch * num, m)
        z_smp, mu, sigma = self.vae.latent_space_from_code(
            code_inp, self.samples_num)  # (x_n, z_n, m)
        
        surv_func, _ = self.benk(*self._prepare_bg_data(code_inp.shape[0]), mu) # (batch, t_n)
        
        _, surv_steps = self.benk(*self._prepare_bg_data(code_inp.shape[0] * self.samples_num), z_smp.flatten(end_dim=-2)) # (x_n * z_n, t_n)
        surv_steps = surv_steps.reshape(code_inp.shape[0], self.samples_num, -1) # (x_n, z_n, t_n)
        
        E_T = self.calc_exp_time(surv_func)  # (x_n)
        
        mu = mu[:, None, :].expand(-1, z_smp.shape[1], -1)
        sigma = sigma[:, None, :].expand(-1, z_smp.shape[1], -1)
        pdf = self.calc_pdf(z_smp.flatten(end_dim=-2), mu.flatten(end_dim=-2), sigma.flatten(end_dim=-2)).reshape(code_inp.shape[0], self.samples_num)  # (x_n, z_n)
        
        step = self.smoother(surv_steps, self.T_bg, E_T, True)
    
        pi = self.pi_head(step, pdf, E_T)
        
        z_est = torch.sum(pi[..., None] * z_smp, dim=1)  # (x_n, m)

        x_est = self.vae.decoder(z_est)
        return x_est.reshape(out_shape), E_T.reshape(x_1.shape[0], num) 
        

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x_list = []
            t_list = []
            X = TensorDataset(torch.from_numpy(x).to('cpu'))
            batch = x.shape[0] if self.batch_load is None else self.batch_load
            dl = DataLoader(X, batch, False)
            for x_b in dl:
                X_b = x_b[0].to(DEVICE)
                x_est, E_T, *_ = self(X_b)
                x_list.append(x_est.cpu().numpy())
                t_list.append(E_T.cpu().numpy())
            X = np.concatenate(x_list, axis=0)
            T = np.concatenate(t_list, axis=0)
            if self.cens_model:
                D_proba = self.cens_model.predict_proba(np.concatenate((X, T[:, None]), -1))[:, 1]
            else:
                D_proba = self.uncens_part
            D = np.random.binomial(1, D_proba, x.shape[0])
            return X, T, D
        
    def predict_trajectory(self, x: np.ndarray, points_num: int, t_min: float, t_max: float)  -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            X = torch.from_numpy(x).to(DEVICE)
            x_est, T, = self.forward_trajectory(X, points_num, t_min, t_max, True)
            return x_est.cpu().numpy(), T.cpu().numpy()

    def predict_interpolate(self, x_1: np.ndarray, x_2: np.ndarray, num: int = 100):
        with torch.no_grad():
            X_1, X_2 = torch.from_numpy(x_1).to(DEVICE), torch.from_numpy(x_2).to(DEVICE)
            x, T = self.interpolate_code(X_1, X_2, num)
            return x.cpu().numpy(), T.cpu().numpy()
        
    def sample_data(self, samples_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            code = torch.normal(0, self.vae.sigma_z, (samples_num, self.vae.latent_dim), 
                                device=DEVICE, dtype=torch.get_default_dtype())
            x_est = self.vae.decoder(code)
            mu = self.vae.encoder(x_est)
            surv_func, surv_steps = self.benk(*self._prepare_bg_data(samples_num), mu) # (batch, t_n)
            E_T = self.gen_exp_time(surv_steps) # (x_n)
            X = x_est.cpu().numpy()
            T = E_T.cpu().numpy()
            
            # x_proto = self.vae.decoder(code)
            # x_list = []
            # t_list = []
            # X = TensorDataset(x_proto)
            # batch = samples_num if self.batch_load is None else self.batch_load
            # dl = DataLoader(X, batch, False)
            # for x_b in dl:
            #     x_est, E_T, *_ = self(x_b[0])
            #     x_list.append(x_est.cpu().numpy())
            #     t_list.append(E_T.cpu().numpy())
            # X = np.concatenate(x_list, axis=0)
            # T = np.concatenate(t_list, axis=0)
            
        if self.cens_model:
            D_proba = self.cens_model.predict_proba(np.concatenate((X, T[:, None]), axis=-1))[:, 1]
        else:
            D_proba = self.uncens_part
        D = np.random.binomial(1, D_proba, samples_num)
        return X, T, D
