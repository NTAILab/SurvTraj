import torch
from .vae import VAE
from .beran import Beran
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from typing import Dict, Tuple, Optional
import time
import numpy as np
from numba import njit, bool_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .smoother import SoftminStepSmoother
import copy

@njit
def dataset_generator(X, T, D, background_size, batch_n):
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

# replaces 0's with the previous difference
@njit
def calc_t_diff_proba(t_labels):
    result = np.empty(t_labels.shape[0], dtype=np.float64)
    for i in range(1, t_labels.shape[0]):
        last_diff = t_labels[-i] - t_labels[-i - 1]
        if last_diff > 1e-8:
            break
    result[-1] = last_diff
    for i in range(1, t_labels.shape[0]):
        cur_diff = t_labels[-i] - t_labels[-i - 1]
        if cur_diff > 1e-8:
            result[-i - 1] = cur_diff
            last_diff = cur_diff
        else:
            result[-i - 1] = last_diff
    return result


# smoothed c index, should be maximized
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
                 c_ind_weight: float = 1.0,
                 vae_weight: float = 1.0,
                 traj_weight: float = 1.0,
                 likelihood_weight: float = 1.0,
                 c_ind_temp: float = 1.0,
                 gumbel_tau: float = 1.0,
                 train_bg_part: float = 0.6,
                 traj_penalty_points_n: int = 10,
                 cens_clf_model = None,
                 batch_load: Optional[int] = None,
                 patience: int = 10, 
                 device: str = 'cpu') -> None:
        super().__init__()
        self.device = torch.device(device)
        self.vae = VAE(device=self.device, **vae_kw)
        self.samples_num = samples_num
        self.beran = Beran(self.device)
        self.batch_num = batch_num
        self.epochs = epochs

        self.lr_rate = lr_rate
        self.traj_pnl_p_n = traj_penalty_points_n
        self.vae_loss = self.vae.mmd_loss
        self.beran_loss = c_ind_loss
        self.c_ind_temp = c_ind_temp
        self.c_ind_w = c_ind_weight
        self.vae_w = vae_weight
        self.traj_w = traj_weight
        self.llh_w = likelihood_weight
        self.batch_load = batch_load
        self.patience = patience
        
        self.smoother = SoftminStepSmoother(self.device)
        if cens_clf_model is not None:
            assert hasattr(cens_clf_model, 'predict_proba')
            assert hasattr(cens_clf_model, 'fit')
        self.cens_model = cens_clf_model
        self.uncens_part = None
        self.train_bg_part = train_bg_part
        self.gumbel_tau = gumbel_tau
        self.T_std = 1
        self.km_times = None
        self.km_hist = None
        self.km_proba = None
        
    def _init_km(self, t: np.ndarray, d: np.ndarray):
        t_km, sf_km = kaplan_meier_estimator(d, t)
        self.km_times = t_km
        self.km_hist = np.empty_like(sf_km)
        self.km_hist[1:] = sf_km[:-1] - sf_km[1:]
        self.km_hist[0] = 1 - sf_km[1]
        diff = np.empty_like(self.km_times)
        diff[:-1] = self.km_times[1:] - self.km_times[:-1]
        diff[-1] = diff[-2]
        self.km_proba = self.km_hist / diff
        
        self.km_times = self.np2torch(self.km_times)
        self.km_hist = self.np2torch(self.km_hist)
        self.km_proba = self.np2torch(self.km_proba)
        
    def _lazy_init(self, x: np.ndarray, y: np.recarray):
        self.T_std = np.std(y['time'])
        self.T_mean = np.mean(y['time'])
        self.vae.dec_dim = self.vae.latent_dim + 1
        self.vae._lazy_init(x)
        self._init_km(y['time'], y['censor'])

    def _set_background(self, X: torch.Tensor, T: torch.Tensor, D: torch.Tensor):
        self.T_bg, sort_args = torch.sort(T)
        self.X_bg = X[sort_args]
        self.D_bg = D[sort_args]
        # T_diff is used for the integral computation, SO:
        # the same time labels will have the same diff metric:
        # (1, 2, 2, 3) -> (1, 1, 1, 1)
        t_diff = calc_t_diff_proba(self.T_bg.cpu().numpy())
        assert np.all(t_diff > 0)
        self.T_diff_int = self.T_bg[1:] - self.T_bg[:-1]
        self.T_diff_prob = self.np2torch(t_diff)
        self.zero_mask = self.T_diff_prob < 1e-8
        
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), self.lr_rate)
    
    def np2torch(self, arr: np.ndarray, 
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = self.device
        return torch.from_numpy(arr).type(dtype).to(device)
    
    def calc_beran_traj_val(self, x: torch.Tensor, 
                             t_min: torch.Tensor, t_event: torch.Tensor,
                             points_n: int) -> torch.Tensor:
        assert points_n > 1
        t = (t_event - t_min) / (points_n - 1)
        assert torch.all(t >= 0)
        t = t[:, None].repeat(1, points_n)
        t[:, 0] = t_min
        t = torch.cumsum(t, dim=-1)
        mu, sigma = self.vae.get_mu_sigma(x)
        xi_z = self.calc_prototype(mu, sigma, t, True) # (batch, points_n, z_dim)
        # dec_points = torch.concat((xi_z, t[..., None]), -1).flatten(end_dim=-2)
        # mu, _ = self.vae.get_mu_sigma(self.vae.decoder(dec_points))
        surv_func, _ = self.beran(*self._prepare_bg_data(xi_z.shape[0] * xi_z.shape[1]), xi_z.flatten(end_dim=-2))
        E_T = self.calc_exp_time(surv_func)
        # D = torch.ones(xi_z.shape[0] * xi_z.shape[1], device=self.device)
        # c_ind = c_ind_loss(t.ravel(), D, E_T, self.c_ind_temp)
        c_ind, *_ = concordance_index_censored(np.ones(xi_z.shape[0] * xi_z.shape[1], dtype=bool), 
                                               t.ravel().cpu().numpy(), -E_T.cpu().numpy())
        mae = torch.mean(torch.abs(t.ravel() - E_T))
        return c_ind, mae.item()
    
    # mean c ind per trajectories
    def c_ind_traj_loss(self, T_true, T_calc, sigma_temp):
        assert T_true.ndim == 2 
        assert T_calc.ndim == 2
        T_diff = T_true[:, :, None] - T_true[:, None, :]
        T_mask = T_diff < 0
        sigma = torch.nn.functional.sigmoid(sigma_temp * (T_calc[:, None, :] - T_calc[:, :, None]))
        C = torch.sum(T_mask * sigma, dim=(1, 2)) / torch.sum(T_mask, dim=(1, 2))
        C_mean = torch.mean(C)
        return C_mean
    
    def calc_beran_traj_loss(self, x: torch.Tensor, 
                             t_min: torch.Tensor, t_event: torch.Tensor,
                             points_n: int) -> torch.Tensor:
        assert points_n > 1
        t = (t_event - t_min) / (points_n - 1)
        assert torch.all(t >= 0)
        t = t[:, None].repeat(1, points_n)
        t[:, 0] = t_min
        t = torch.cumsum(t, dim=-1)
        mu, sigma = self.vae.get_mu_sigma(x)
        xi_z = self.calc_prototype(mu, sigma, t, True) # (batch, points_n, z_dim)
        # dec_points = torch.concat((xi_z, t[..., None]), -1).flatten(end_dim=-2)
        # mu, _ = self.vae.get_mu_sigma(self.vae.decoder(dec_points))
        surv_func, _ = self.beran(*self._prepare_bg_data(xi_z.shape[0] * xi_z.shape[1]), xi_z.flatten(end_dim=-2))
        E_T = self.calc_exp_time(surv_func).reshape(xi_z.shape[0], xi_z.shape[1])
        # D = torch.ones(xi_z.shape[0] * xi_z.shape[1], device=self.device)
        c_ind = self.c_ind_traj_loss(t, E_T, self.c_ind_temp)
        # c_ind_list = []
        # d = torch.ones_like(t[0, :])
        # for i in range(x.shape[0]):
        #     c_ind_list.append(c_ind_loss(t[i, :], d, E_T[i, :], self.c_ind_temp))
        # c_ind_mean = torch.mean(torch.stack(c_ind_list))
        # c_ind_full = c_ind_loss(t.ravel(), torch.ones_like(E_T.ravel()), E_T.ravel(), self.c_ind_temp)
        # c_ind_hard, *_ = concordance_index_censored(np.ones(xi_z.shape[0] * xi_z.shape[1], dtype=bool), 
        #                                        t.ravel().detach().cpu().numpy(), -E_T.detach().ravel().cpu().numpy())
        # mae = torch.mean(torch.abs(t.ravel() - E_T.ravel()))
        return c_ind
    
    def calc_likelihood(self, x: torch.Tensor, t: torch.Tensor):
        self.beran.kernel.bandwidth.requires_grad = False
        # self.smoother.bandwidth.requires_grad = False
        mu, sigma = self.vae.get_mu_sigma(x)
                
        batch_len, dim = mu.shape
        z_i_n = self.samples_num
        z_i = self.vae.gen_samples(mu, sigma, z_i_n).reshape(-1, dim) # (batch * z_i_n, dim)
        mu_expanded = mu[:, None, :].expand(-1, z_i_n, -1).reshape(z_i.shape) # (batch * z_i_n, dim)
        sigma_expanded = sigma[:, None, :].expand(-1, z_i_n, -1).reshape(z_i.shape) # (batch * z_i_n, dim)
        pi_z = self.calc_pdf(z_i, mu_expanded, sigma_expanded) # (batch * z_i_n)
        
        mu_bg, delta_bg = self._prepare_bg_data(z_i.shape[0])
        _, surv_steps = self.beran(mu_bg.detach(), delta_bg.detach(), z_i) # (batch * z_i_n, t_n)
        pi_t_z = self.surv_steps_to_proba(surv_steps) # (batch * z_i_n, t_n)
        bg_times = self.T_bg
        pi_t_z = self.smoother(pi_t_z, bg_times, t, self.samples_num) # (batch, z_i_n, p_n) or (batch, z_i_n)
        pi_z = torch.reshape(pi_z, (batch_len, self.samples_num)) # (batch, z_i_n)
        z_i = torch.reshape(z_i, (batch_len, self.samples_num, dim))
        weights = self.calc_xi_weights(pi_t_z, pi_z) # same as input, weighted within 1'st dim
        xi_z = torch.sum(weights[..., None] * z_i, dim=1) # (batch, z_dim) or (batch, p_n, z_dim)
        
        mu_bg, delta_bg = self._prepare_bg_data(xi_z.shape[0])
        _, proba = self.beran(mu_bg.detach(), delta_bg.detach(), xi_z)
        step_idx = torch.searchsorted(self.T_bg, t).clamp_max_(proba.shape[1] - 1)
        steps = torch.take_along_dim(proba, step_idx[:, None], dim=1)
        steps[steps < 1e-13] = 1
        logs = torch.log(steps)
        self.beran.kernel.bandwidth.requires_grad = True
        # self.smoother.bandwidth.requires_grad = True
        # return torch.mean(logs)
        
        km_step_idx = torch.searchsorted(self.km_times, t).clamp_max_(self.km_proba.shape[0] - 1)
        km_proba = torch.take_along_dim(self.km_proba, km_step_idx, 0)
        metric_weights = torch.nn.functional.softmin(km_proba, 0)
        return torch.sum(metric_weights * logs[:, 0])
        
        
    def fit(self, x: np.ndarray, y: np.recarray, 
            val_set: Optional[Tuple[np.ndarray, np.recarray]]=None,
            log_dir:Optional[str]=None) -> 'SurvivalMixup':
        assert x.shape[0] == y.shape[0]
        self._lazy_init(x, y)
        optimizer = self._get_optimizer()
        start_time = time.time()
        self.train()
        
        if log_dir is not None:
            summary_writer = SummaryWriter(log_dir)
        
        cur_patience = 0
        best_val_loss = 0

        back_size = int(self.train_bg_part * x.shape[0])
        print('Task volume:', x.shape[0] - back_size)
        print('Background volume:', back_size)
        sub_batch_len = self.batch_load if self.batch_load is not None else x.shape[0] - back_size
        
        train_T = y['time'].copy()
        train_D = y['censor'].copy()
        self.uncens_part = np.sum(train_D) / train_D.shape[0]
        X_full_tens = self.np2torch(x)
        T_full_tens = self.np2torch(train_T)
        D_full_tens = self.np2torch(train_D, torch.int)
        traj_min_time = torch.min(T_full_tens)
        traj_max_time = torch.max(T_full_tens)
        
        for e in range(1, self.epochs + 1):
            cum_loss, cum_x_loss, cum_reg_loss, cum_benk_loss = 0, 0, 0, 0
            cum_traj_c_val, cum_traj_mae = 0, 0
            cum_traj_c_loss, cum_llh_loss = 0, 0
            i = 0
            data = dataset_generator(x, train_T, train_D, back_size, self.batch_num)
            X_back = self.np2torch(data[0])
            T_back = self.np2torch(data[1])
            D_back = self.np2torch(data[2], torch.int)
            X_target = self.np2torch(data[3], device='cpu')
            T_target = self.np2torch(data[4], device='cpu')
            D_target = self.np2torch(data[5], torch.int, device='cpu')
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
                traj_c_ind_val, traj_mae_val = 0, 0
                traj_c_ind_loss = 0
                likelihood = 0
                
                for x_t_b, t_t_b, d_t_b in target_loader:
                    x_t_b, t_t_b, d_t_b = x_t_b.to(self.device), t_t_b.to(self.device), d_t_b.to(self.device)
                    x_est, E_T, T_gen, z = self(x_t_b)

                    benk_loss += self.beran_loss(t_t_b, d_t_b, E_T, self.c_ind_temp)
                    
                    uncens_mask = d_t_b == 1
                    x_uncens = x_t_b[uncens_mask]
                    t_uncens = t_t_b[uncens_mask]
                    t_min = traj_min_time.expand(x_t_b.shape[0])
                    t_max = traj_max_time.expand(x_t_b.shape[0])
                    traj_c_ind_loss += self.calc_beran_traj_loss(x_t_b, t_min, t_max, self.traj_pnl_p_n)
                
                    cur_vae_loss, cur_x_loss, cur_regularizer = self.vae_loss(x_t_b, z, x_est)
                    
                    likelihood += self.calc_likelihood(x_uncens, t_uncens)               
                    vae_loss += cur_vae_loss
                    x_loss += cur_x_loss
                    regularizer += cur_regularizer
                    
                    with torch.no_grad():
                        x_val_len = min(x_t_b.shape[0], 10)
                        t_min = traj_min_time.expand(x_val_len)
                        t_max = traj_max_time.expand(x_val_len)
                        x_val_check = x_t_b[:x_val_len]
                        # t_min = traj_min_time.expand(1)
                        # t_max = traj_max_time.expand(1)
                        # x_val_check = x_t_b[0, None, ...]
                        tr_c, tr_mae = self.calc_beran_traj_val(x_val_check, t_min, t_max, self.traj_pnl_p_n)
                        traj_c_ind_val += tr_c
                        traj_mae_val += tr_mae
                
                tl_len = len(target_loader)
                benk_loss /= tl_len
                vae_loss /= tl_len
                x_loss /= tl_len
                regularizer /= tl_len
                traj_c_ind_val /= tl_len
                traj_mae_val /= tl_len
                traj_c_ind_loss /= tl_len
                
                loss = -self.c_ind_w * benk_loss
                loss += -self.traj_w * traj_c_ind_loss
                loss += self.vae_w * vae_loss
                loss += -self.llh_w * likelihood
                # loss = self.c_ind_w * (-benk_loss - traj_c_ind_loss) + (1 - self.c_ind_w) * vae_loss

                loss.backward()

                optimizer.step()

                cum_loss += loss.item()
                cum_x_loss += x_loss
                cum_reg_loss += regularizer
                cum_benk_loss += benk_loss.item()
                cum_traj_c_val += traj_c_ind_val
                cum_traj_mae += traj_mae_val
                cum_traj_c_loss += traj_c_ind_loss.item()
                cum_llh_loss += likelihood.item()
                
                i += 1
                epoch_metrics = {
                    'Loss': cum_loss / i,
                    'Recon': cum_x_loss / i,
                    'Regul': cum_reg_loss / i,
                    'C ind': cum_benk_loss / i,
                    'Traj C': cum_traj_c_loss / i,
                    'Likelihood': cum_llh_loss / i,
                    'Traj C val': cum_traj_c_val / i,
                    'Traj MAE': cum_traj_mae / i,
                }
                prog_bar.set_postfix(epoch_metrics)
            if log_dir is not None:
                summary_writer.add_scalars('train_metrics', epoch_metrics, e)
            if val_set is not None:
                cur_patience += 1
                self._set_background(X_full_tens, T_full_tens, D_full_tens)
                val_loss = self.score(*val_set)
                if val_loss >= best_val_loss:
                    best_val_loss = val_loss
                    weights = copy.deepcopy(self.state_dict())
                    cur_patience = 0
                print(f'Val C-index: {round(val_loss, 5)}, patience: {cur_patience}')
                if log_dir is not None:
                    summary_writer.add_scalars('val_metric', val_loss, e)
                if cur_patience >= self.patience:
                    print('Early stopping!')
                    self.load_state_dict(weights)
                    break
        self._set_background(X_full_tens, T_full_tens, D_full_tens)
        time_elapsed = time.time() - start_time
        print('Training time:', round(time_elapsed, 1), 's.')
        if log_dir is not None:
            summary_writer.close()
        self.eval()
        if self.cens_model is not None:
            self.fit_cens_model(X_full_tens, T_full_tens, train_D)
        return self

    def fit_cens_model(self, X: torch.Tensor, T: torch.Tensor, d: np.ndarray) -> None:
        with torch.no_grad():
            mu, sigma = self.vae.get_mu_sigma(X)
            # xi_z = self.calc_prototype(mu, sigma, T)
            # xi_z = xi_z.cpu().numpy()
            t = T.cpu().numpy()
        train_data = np.concatenate((mu.cpu().numpy(), t[:, None]), axis=-1)
        self.cens_model.fit(train_data, d.astype(np.int0))
        
    def _prepare_bg_data(self, batch_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, _ = self.vae.get_mu_sigma(self.X_bg)
        mu = mu[None, ...].expand(batch_num, -1, -1)
        D_bg = self.D_bg[None, :].expand(batch_num, -1)
        return mu, D_bg

    def calc_exp_time(self, surv_func: torch.Tensor) -> torch.Tensor:
        integral = self.T_bg[None, 0] + \
            torch.sum(surv_func[:, :-1] * self.T_diff_int[None, :], dim=-1)
        return integral # (batch)
    
    def gen_evt_time(self, surv_steps: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(surv_steps, 1e-10)
        softmax = torch.nn.functional.gumbel_softmax(logits, self.gumbel_tau, dim=-1, hard=True)
        T_gen = torch.sum(time[None, :] * softmax, -1)
        return T_gen
    
    def calc_pdf(self, x: torch.Tensor, mu: torch.Tensor, var_log: torch.Tensor) -> torch.Tensor:
        x_cnt = x - mu
        sigma = torch.exp(var_log / 2) # (x_n, m)
        det_sqrt = torch.prod(sigma, dim=-1) # (x_n)
        cov_mtx_inv = 1 / torch.exp(var_log) # (x_n, m)
        return 1 / (np.power(2 * np.pi, x.shape[-1] / 2) * det_sqrt) *\
            torch.exp(-1 / 2 * torch.sum(x_cnt * cov_mtx_inv * x_cnt, dim=-1))
    
    # x, mu: 2 dim
    def calc_pdf_const_sigma(self, x: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
        m = x.shape[-1]
        x_unb = x - mu
        denom = np.power(2 * np.pi, m / 2) * np.power(sigma, m)
        sigma_inv = torch.ones_like(x_unb) / (sigma ** 2)
        prob = 1 / denom * torch.exp(-0.5 * torch.sum(x_unb * sigma_inv * x_unb, dim=-1))
        return prob
    
    def calc_xi_weights(self, pi_t_z: torch.Tensor, pi_z: torch.Tensor) -> torch.Tensor:       
        pi = pi_t_z * pi_z # (x_n, z_n, p_n)
        sum = torch.sum(pi, dim=1, keepdim=True).broadcast_to(pi.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        pi = pi / sum  # (x_n, z_n)
        pi[bad_idx] = 0
        return pi
            
    def surv_steps_to_proba(self, surv_steps: torch.Tensor) -> torch.Tensor:
        t_diff_corr = self.T_diff_prob.clone()
        mask = self.zero_mask.clone()
        t_diff_corr[mask] = 1
        t_diff_corr = t_diff_corr[None, :]
        mask = mask[None, :]
        if surv_steps.ndim == 3:
            t_diff_corr = t_diff_corr[None, ...]
            mask = mask[None, ...]
        # proba = surv_steps / torch.sum(surv_steps, -1, keepdim=True)
        proba = surv_steps
        proba = proba / t_diff_corr
        mask = mask.broadcast_to(proba.shape)
        proba[mask] = 0
        # assert not torch.any(torch.isnan(proba))
        return proba
    
    # calculate the survival function and histogram based on it
    # 2 dim and 3 dim tensors are proceeded
    # background is (x, delta) tuple, sorted according to the time to event
    def calc_beran(self, z: torch.Tensor, 
                   background: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim == 3:
            dim2 = z.shape[1]
            z = z.flatten(end_dim=-1)
        sf, pi = self.beran(*background, z)
        if z.ndim == 3:
            sf = sf.reshape((-1, dim2, sf.shape[-1]))
            pi = pi.reshape((-1, dim2, pi.shape[-1]))
        return sf, pi
    
    # calc xi_z(t)
    # t is 1 dim or 2 dim.
    # f_single_set - generate 1 set of z_i in case of trajectory or multiple sets.
    # output is 2 dim (batch, z_dim) or 3 dim (batch, traj_n, z_dim) tensor.
    def calc_prototype(self, mu: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor,
                       f_single_set: bool = True, f_return_z_i: bool = False) -> torch.Tensor:
        f_trajectory = t.ndim == 2
        tr_len = t.shape[-1]
        batch_len, dim = mu.shape
        if f_trajectory and not f_single_set:
            z_i_n = self.samples_num * tr_len
        else:
            z_i_n = self.samples_num
        z_i = self.vae.gen_samples(mu, sigma, z_i_n).reshape(-1, dim) # (batch * z_i_n, dim)
        mu_expanded = mu[:, None, :].expand(-1, z_i_n, -1).reshape(z_i.shape) # (batch * z_i_n, dim)
        sigma_expanded = sigma[:, None, :].expand(-1, z_i_n, -1).reshape(z_i.shape) # (batch * z_i_n, dim)
        pi_z = self.calc_pdf(z_i, mu_expanded, sigma_expanded) # (batch * z_i_n)
        
        _, surv_steps = self.beran(*self._prepare_bg_data(z_i.shape[0]), z_i) # (batch * z_i_n, t_n)
        pi_t_z = self.surv_steps_to_proba(surv_steps) # (batch * z_i_n, t_n)
        # assert not torch.any(torch.isnan(pi_t_z))
        bg_times = self.T_bg
        pi_t_z = self.smoother(pi_t_z, bg_times, t, self.samples_num) # (batch, z_i_n, p_n) or (batch, z_i_n)
        # assert not torch.any(torch.isnan(pi_t_z))
        if f_trajectory:
            if f_single_set:
                pi_z = torch.reshape(pi_z, (batch_len, self.samples_num, 1)) # (batch, z_i_n, 1)
                z_i = torch.reshape(z_i, (batch_len, self.samples_num, 1, dim))
            else:
                pi_z = torch.reshape(pi_z, (batch_len, self.samples_num, tr_len)) # (batch, z_i_n, p_n)
                z_i = torch.reshape(z_i, (batch_len, self.samples_num, tr_len, dim))
        else:
            pi_z = torch.reshape(pi_z, (batch_len, self.samples_num)) # (batch, z_i_n)
            z_i = torch.reshape(z_i, (batch_len, self.samples_num, dim))
        weights = self.calc_xi_weights(pi_t_z, pi_z) # same as input, weighted within 1'st dim
        z_prot = torch.sum(weights[..., None] * z_i, dim=1) # (batch, z_dim) or (batch, p_n, z_dim)
        # assert not torch.any(torch.isnan(z_prot))
        if f_return_z_i:
            assert not f_trajectory
            to_return = (z_prot, z_i[:, 0, :])
        else:
            to_return = z_prot
        return to_return
        
    # from x to its reconstrucrion and estimated time
    def forward(self, x: torch.Tensor):
        mu, sigma = self.vae.get_mu_sigma(x)
        surv_func, surv_steps = self.beran(*self._prepare_bg_data(mu.shape[0]), mu) # (batch, t_n)
        T_gen = self.gen_evt_time(surv_steps, self.T_bg) # (x_n)
        E_T = self.calc_exp_time(surv_func)
        xi_z, z_i = self.calc_prototype(mu, sigma, T_gen, f_return_z_i=True)
        T_feat = ((T_gen - self.T_mean) / self.T_std)[:, None]
        x_est = self.vae.decoder(torch.concat((xi_z, T_feat), dim=-1))
        # assert not torch.any(torch.isnan(x_est))
        return x_est, E_T, T_gen, z_i
    
    def forward_trajectory(self, x: torch.Tensor, t: torch.Tensor, f_single_samples_set=True):
        assert t.ndim == 2
        mu, sigma = self.vae.get_mu_sigma(x)
        xi_z = self.calc_prototype(mu, sigma, t, f_single_samples_set)
        T_feat = ((t - self.T_mean) / self.T_std)[..., None]
        x_est = self.vae.decoder(torch.concat((xi_z, T_feat), dim=-1))
        return x_est

    # conditional sampling
    def predict_recon(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            x_list = []
            # xi_list = []
            mu_list = []
            t_gen_list = []
            X = TensorDataset(self.np2torch(x, device='cpu'))
            batch = x.shape[0] if self.batch_load is None else self.batch_load
            dl = DataLoader(X, batch, False)
            for x_b in dl:
                X_b = x_b[0].to(self.device)
                
                mu, sigma = self.vae.get_mu_sigma(X_b)
                _, surv_steps = self.beran(*self._prepare_bg_data(mu.shape[0]), mu) # (batch, t_n)
                T_gen = self.gen_evt_time(surv_steps, self.T_bg) # (x_n)
                xi_z = self.calc_prototype(mu, sigma, T_gen)
                T_feat = ((T_gen - self.T_mean) / self.T_std)[:, None]
                x_est = self.vae.decoder(torch.concat((xi_z, T_feat), dim=-1))
                
                x_list.append(x_est.cpu().numpy())
                # xi_list.append(xi_z.cpu().numpy())
                mu_list.append(mu.cpu().numpy())
                t_gen_list.append(T_gen.cpu().numpy())
            X = np.concatenate(x_list, axis=0)
            T_gen = np.concatenate(t_gen_list, axis=0)
            # Xi = np.concatenate(xi_list, axis=0)
            Mu = np.concatenate(mu_list, axis=0)
            if self.cens_model is not None:
                D_proba = self.cens_model.predict_proba(
                    np.concatenate((Mu, T_gen[:, None]), -1))[:, 1]
            else:
                D_proba = self.uncens_part
            D = np.random.binomial(1, D_proba, x.shape[0])
            return X, T_gen, D
    
    # predict xi_x in certain time for each x
    def predict_in_time(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        assert t.ndim == 1 and t.shape[0] == x.shape[0]
        with torch.no_grad():
            X = self.np2torch(x)
            T = self.np2torch(t)
            mu, sigma = self.vae.get_mu_sigma(X)
            xi_z = self.calc_prototype(mu, sigma, T)
            T_feat = ((T - self.T_mean) / self.T_std)[:, None]
            x_est = self.vae.decoder(torch.concat((xi_z, T_feat), dim=-1))
            return x_est.cpu().numpy()
    
    # t is 2 dim, unique vector for each x
    def predict_trajectory(self, x: np.ndarray, t: np.ndarray, multi_sampling: bool=False)  -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            X = self.np2torch(x)
            T = self.np2torch(t)
            x_est = self.forward_trajectory(X, T, not multi_sampling)
            return x_est.cpu().numpy()
        
    def predict_exp_time(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = self.np2torch(x)
            mu, _ = self.vae.get_mu_sigma(X)
            surv_func, _ = self.beran(*self._prepare_bg_data(mu.shape[0]), mu) # (batch, t_n)
            E_T = self.calc_exp_time(surv_func)
            return E_T.cpu().numpy()
        
    def generate_apriori_t(self, num: int) -> torch.Tensor:
        steps = self.km_hist[None, :].expand(num, -1)
        T = self.gen_evt_time(steps, self.km_times)
        return T
        
    def sample_data(self, samples_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            T_gen = self.generate_apriori_t(samples_num)
            mu = torch.normal(0, 1, (samples_num, self.vae.latent_dim), 
                             dtype=torch.get_default_dtype(), device=self.device)
            sigma = self.vae.sigma_nn(mu)
            xi_z = self.calc_prototype(mu, sigma, T_gen)
            T_feat = ((T_gen - self.T_mean) / self.T_std)[:, None]
            x_est = self.vae.decoder(torch.concat((xi_z, T_feat), dim=-1))
            T_gen = T_gen.cpu().numpy()
            x_est = x_est.cpu().numpy()
            if self.cens_model is not None:
                cls_features = np.concatenate((xi_z.cpu().numpy(), T_gen[:, None]), axis=1)
                D_proba = self.cens_model.predict_proba(cls_features)[:, 1]
            else:
                D_proba = self.uncens_part
            D = np.random.binomial(1, D_proba, samples_num)
            return x_est, T_gen, D

    def score(self, x: np.ndarray, y: np.recarray) -> float:
        E_T = self.predict_exp_time(x)
        c_ind, *_ = concordance_index_censored(y['censor'], y['time'], -E_T)
        return c_ind
        