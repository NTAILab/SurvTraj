import torch
import numpy as np
from typing import Tuple, Optional
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

class VAE(torch.nn.Module):
    def _get_encoder(self, dim_in: int, dim_out: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 10 * dim_in),
            torch.nn.ReLU(),
            torch.nn.Linear(10 * dim_in, dim_out),
            torch.nn.Tanh(),
        )

    def _get_mu(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 2 * dim_latent),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * dim_latent, dim_latent)
        )

    def _get_sigma(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 2 * dim_latent),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * dim_latent, dim_latent)
        )

    def _get_decoder(self, dim_latent: int, dim_out: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_latent, 10 * dim_latent),
            torch.nn.ReLU(),
            torch.nn.Linear(10 * dim_latent, 4 * dim_latent),
            torch.nn.Tanh(),
            torch.nn.Linear(4 * dim_latent, dim_out)
        )

    def _get_optimizer(self, lr_rate: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr_rate)

    def __init__(self, latent_dim: int, kl_coef: float, lr_rate: float, batch_num: int, epochs_num: int) -> None:
        super().__init__()
        self.lr_rate = lr_rate
        self.kl_coef = kl_coef
        self.batch_num = batch_num
        self.epochs_num = epochs_num
        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None
        self.mu_nn = None
        self.sigma_nn = None
        self.optmizer = None

    def _lazy_init(self, x: np.ndarray) -> None:
        dim = x.shape[-1]
        self.dim = dim

        self.encoder = self._get_encoder(dim + 1, self.latent_dim)
        self.mu_nn = self._get_mu(self.latent_dim, self.latent_dim)
        self.sigma_nn = self._get_sigma(self.latent_dim, self.latent_dim)
        self.decoder = self._get_decoder(self.latent_dim + 1, dim)
        self.optmizer = self._get_optimizer(self.lr_rate)

    def loss_fn(self, true: torch.Tensor, recon: torch.Tensor,
                mu: torch.Tensor, sigma_log: torch.Tensor) -> torch.Tensor:
        mse = torch.mean(torch.sum((true - recon) ** 2, dim=-1))
        kl = 0.5 * torch.mean(torch.sum(torch.exp(sigma_log) + mu ** 2 - sigma_log - 1, dim=-1))
        loss = mse + self.kl_coef * kl
        return loss, mse, kl

    def gen_val_loss(self, x: torch.Tensor, c: torch.Tensor, val_num: int) -> float:
        with torch.no_grad():
            epsilon = torch.randn((val_num, self.latent_dim))
            code = torch.concat((epsilon, c), dim=-1)
            x_val = self.decoder(code)[:, None, :]
            mse_diff = torch.mean((x_val - x[None, ...]) ** 2, dim=-1)
            min = torch.amin(mse_diff, dim=-1)
        return torch.mean(min).item()

    def rec_val_loss(self, x: torch.Tensor, c: torch.Tensor) -> float:
        with torch.no_grad():
            rec, *_ = self(x, c)
            return torch.mean((x - rec) ** 2).item()
        
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == c.shape[0]
        assert c.ndim == 1 or c.ndim == 2 and c.shape[1] == 1
        if c.ndim == 1:
            c = c[:, None]
        return self.encoder(torch.concat((x, c), dim=-1))
    
    def latent_space_from_code(self, code: torch.Tensor, num: int = 1) -> torch.Tensor:
        mu = self.mu_nn(code)
        # actually 2 * log(sigma)
        sigma_log = self.sigma_nn(code)
        mu_out, sigma_out = mu, sigma_log
        epsilon = torch.randn((code.shape[0], num, code.shape[1]))
        if num == 1:
            epsilon = epsilon[:, 0, :]
        else:
            mu = mu[:, None, :].expand(-1, num, -1)
            sigma_log = sigma_log[:, None, :].expand(-1, num, -1)
        latent_space = mu + epsilon * torch.exp(sigma_log / 2)
        return latent_space, mu_out, sigma_out

    def decode(self, latent_space: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == c.shape[0]
        assert c.ndim == 1 or c.ndim == 2 and c.shape[1] == 1
        if c.ndim == 1:
            c = c[:, None]
        full_code = torch.concat((latent_space, c), dim=-1)
        return self.decoder(full_code)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.encode(x, c)
        latent_space, mu, sigma_log = self.latent_space_from_code(code, c)
        reconstruction = self.decode(latent_space, c)
        return reconstruction, mu, sigma_log

    def fit(self, x: np.ndarray, y: np.ndarray,
            gen_val_num: Optional[int] = 100,
            rec_val_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'VAE':
        assert x.ndim == 2
        assert y.ndim == 1
        self._lazy_init(x)
        X = torch.from_numpy(x)
        C = torch.from_numpy(y)[:, None]

        if rec_val_set is not None:
            X_val = torch.from_numpy(rec_val_set[0])
            C_val = torch.from_numpy(rec_val_set[1])

        dataset = TensorDataset(X, C)
        self.train()
        val_kw = dict()
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            cum_loss, cum_mse, cum_kl = 0, 0, 0
            for i, (x_b, c_b) in enumerate(prog_bar):
                self.optmizer.zero_grad()
                x_r, mu, sigma = self(x_b, c_b)
                loss, mse, kl = self.loss_fn(x_b, x_r, mu, sigma)
                loss.backward()
                self.optmizer.step()
                cum_loss += loss.item()
                cum_mse += mse.item()
                cum_kl += kl.item()

                if gen_val_num > 0:
                    C = np.random.choice(y, gen_val_num, True)
                    C = torch.from_numpy(C)[:, None]
                    val_kw['Val_gen'] = self.gen_val_loss(X, C, gen_val_num)

                if rec_val_set is not None:
                    val_kw['Val_rec'] = self.rec_val_loss(X_val, C_val)

                prog_bar.set_postfix(Loss=cum_loss / (i + 1),
                                     MSE=cum_mse / (i + 1),
                                     KL=cum_kl / (i + 1),
                                     **val_kw)
        self.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(x)
            return self(X).cpu().numpy()

    def _sample_without_x(self, num: int, condition: np.ndarray) -> np.ndarray:
        assert condition.shape[0] == num
        with torch.no_grad():
            C = torch.from_numpy(condition)[:, None]
            code = torch.concat((torch.randn((num, self.latent_dim)), C), dim=-1)
            return self.decoder(code).cpu().numpy()

    def sample(self, num: int,
               condition: np.ndarray,
               x: Optional[np.ndarray] = None) -> np.ndarray:
        assert condition.ndim == 1
        if x is None:
            return self._sample_without_x(num, condition)
        assert x.shape[0] == condition.shape[0] or x.shape[0] * num == condition.shape[0]
        with torch.no_grad():
            X = torch.from_numpy(x)
            X = X[:, None, :].expand((-1, num, -1)).flatten(end_dim=-2)
            C = torch.from_numpy(condition)[:, None]
            if x.shape[0] == condition.shape[0]:
                C = C.expand((-1, num))
            C = C.ravel()
            samples, *_ = self(X, C)
            return samples.reshape(-1, num, samples.shape[-1]).cpu().numpy()


if __name__ == '__main__':
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt

    gen_kw = {'n_samples': 100, 'noise': 0.025}
    x, y = make_moons(**gen_kw)
    x_val, y_val = make_moons(**gen_kw)
    x1, x2 = x[y == 0], x[y == 1]
    fig, axis = plt.subplots(1, 1)
    axis.scatter(*x1.T, c='b', s=1.5)
    axis.scatter(*x2.T, c='r', s=1.5)

    model = VAE(1, 0.05, 2e-3, 32, 300)
    model.fit(x, y.astype(np.double), rec_val_set=(x_val, y_val.astype(np.double)))

    cond = np.random.binomial(1, 0.5, 100).astype(np.double)
    rnd = model.sample(100, cond)
    print('rnd shape:', rnd.shape)
    axis.scatter(*rnd.T, marker='*', c='k', s=1.8)

    fig, axis = plt.subplots(1, 1)
    axis.scatter(*x1.T, c='b', s=0.8)
    axis.scatter(*x2.T, c='r', s=0.8)
    exp_points = np.stack([x1[0], x2[0]], axis=0)
    gen_num = 20
    # cond = np.random.binomial(2, 0.5, 40).astype(np.double)
    cond = np.asarray([0.0, 1.0])
    recon = model.sample(20, cond, exp_points)
    print('rec shape:', recon.shape)
    recon1, recon2 = recon[0], recon[1]
    axis.scatter(*recon1.T, c='b', marker='^')
    axis.scatter(*recon2.T, c='r', marker='v')
    axis.scatter(*exp_points.T, c='k', marker='*')

    plt.show()
