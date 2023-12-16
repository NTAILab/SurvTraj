import torch
import numpy as np
from typing import Tuple, Optional
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from . import DEVICE

class VAE(torch.nn.Module):
    def _get_encoder(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            # torch.nn.Linear(2 * dim_latent, 2 * dim_latent),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(48, 48),
            # torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            # torch.nn.Linear(32, 32),
            # torch.nn.Tanh(),
            # torch.nn.Linear(32, 32),
            # torch.nn.Tanh(),
            torch.nn.Linear(32, dim_latent)
        ).to(DEVICE)

    def _get_sigma(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_latent),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_latent, dim_latent)
        ).to(DEVICE)

    def _get_decoder(self, dim_latent: int, dim_out: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_latent, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            # torch.nn.Linear(48, 48),
            # torch.nn.Tanh(),
            # torch.nn.Linear(48, 64),
            # torch.nn.Tanh(),
            # torch.nn.Linear(32, 32),
            # torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, dim_out)
        ).to(DEVICE)

    def __init__(self, latent_dim: int, regular_coef: float, sigma_z: float,
                 encoder_dim: Optional[int]=None, decoder_dim: Optional[int]=None) -> None:
        super().__init__()
        self.reg_coef = regular_coef
        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None
        self.mu_nn = None
        self.sigma_nn = None
        self.sigma_z = sigma_z
        self.enc_dim = encoder_dim
        self.dec_dim = decoder_dim

    def _lazy_init(self, x: np.ndarray) -> None:
        enc_dim = x.shape[-1] if self.enc_dim is None else self.enc_dim
        dec_dim = self.latent_dim if self.dec_dim is None else self.dec_dim
        
        self.encoder = self._get_encoder(enc_dim, self.latent_dim)
        # self.mu_nn = self._get_mu(self.latent_dim, self.latent_dim)
        self.sigma_nn = self._get_sigma(self.latent_dim, self.latent_dim)
        self.decoder = self._get_decoder(dec_dim, x.shape[-1])
        
    def kernel(self, x_diff: torch.Tensor) -> torch.Tensor:
        C = 2 * self.latent_dim * self.sigma_z * self.sigma_z
        return C / (C + torch.sum(torch.pow(x_diff, 2), dim=-1)) # (batch)
    
    def gauss_kernel(self, x_diff: torch.Tensor) -> torch.Tensor:
        sigma = 0.5#2 * self.latent_dim * self.sigma_z * self.sigma_z
        return torch.exp(- torch.sum(torch.pow(x_diff, 2), dim=-1) / sigma ** 2)
        

    def loss_fn(self, true: torch.Tensor, recon: torch.Tensor,
                mu: torch.Tensor, sigma_log: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        mse = torch.mean(torch.sum((true - recon) ** 2, dim=-1))
        kl = 0.5 * torch.mean(torch.sum(torch.exp(sigma_log) + mu ** 2 - sigma_log - 1, dim=-1))
        loss = mse + self.reg_coef * kl
        return loss, mse.item(), kl.item()
    
    def mmd_loss(self, x: torch.Tensor, z_latent: torch.Tensor, 
                 recon: torch.Tensor):
        assert x.shape[0] == recon.shape[0]
        
        z_smp = torch.normal(0, self.sigma_z, z_latent.shape,
                             device=DEVICE, dtype=torch.get_default_dtype())
        
        cost = torch.mean(torch.sum(torch.pow(x - recon, 2), dim=-1))
        
        kernel = self.kernel
        z_smp_diff_mtx = z_smp[None, ...] - z_smp[:, None, :]
        K_z_smp = kernel(z_smp_diff_mtx.flatten(end_dim=-2))
        z_diff_mtx = z_latent[None, ...] - z_latent[:, None, :]
        K_z = kernel(z_diff_mtx.flatten(end_dim=-2))
        z_dis_diff = z_smp[None, ...] - z_latent[:, None, :]
        K_dis = kernel(z_dis_diff.flatten(end_dim=-2))
        
        n = z_latent.shape[0]
        diag_mask = torch.eye(n, dtype=torch.bool, device=DEVICE).ravel()
        K_z_smp = K_z_smp.masked_fill(diag_mask, 0)
        K_z = K_z.masked_fill(diag_mask, 0)
        
        quad_coef = self.reg_coef / (n * (n - 1))
        full_quad_coef = 2 * self.reg_coef / (n * n)
        
        regularizer = quad_coef * torch.sum(K_z_smp) + quad_coef * torch.sum(K_z) \
            - full_quad_coef * torch.sum(K_dis)
        loss = cost + regularizer
        return loss, cost.item(), regularizer.item()
        
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
    
    def latent_space_from_code(self, code: torch.Tensor, num: int = 1) -> torch.Tensor:
        mu = code #self.mu_nn(code)
        # actually 2 * log(sigma)
        sigma_log = self.sigma_nn(code)
        mu_out, sigma_out = mu, sigma_log
        epsilon = torch.normal(0, self.sigma_z, (code.shape[0], num, code.shape[1]),
                               device=DEVICE, dtype=torch.get_default_dtype())
        if num == 1:
            epsilon = epsilon[:, 0, :]
        else:
            mu = mu[:, None, :].expand(-1, num, -1)
            sigma_log = sigma_log[:, None, :].expand(-1, num, -1)
        latent_space = mu + epsilon * torch.exp(sigma_log / 2)
        return latent_space, mu_out, sigma_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        latent_space, mu, sigma_log = self.latent_space_from_code(code)
        reconstruction = self.decoder(latent_space)
        return reconstruction, mu, sigma_log
