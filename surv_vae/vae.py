import torch
import numpy as np
from typing import Tuple, Optional

class VAE(torch.nn.Module):
    # dim_latent is dim(Z)
    # encoder gives mean and log of std
    def _get_encoder(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, dim_latent)
        ).to(self.device)


    def _get_decoder(self, dim_latent: int, dim_out: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_latent, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, dim_out)
        ).to(self.device)
        
    
    def _get_sigma(self, dim_in: int, dim_latent: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 2 * dim_latent),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * dim_latent, dim_latent)
        ).to(self.device)

    def __init__(self, latent_dim: int, regular_coef: float, sigma_z: float, device: torch.device,
                 encoder_dim: Optional[int]=None, decoder_dim: Optional[int]=None) -> None:
        super().__init__()
        self.device = device
        self.reg_coef = regular_coef
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.sigma_nn = None
        self.sigma_z = sigma_z
        self.enc_dim = encoder_dim
        self.dec_dim = decoder_dim

    def _lazy_init(self, x: np.ndarray) -> None:
        enc_dim = x.shape[-1] if self.enc_dim is None else self.enc_dim
        dec_dim = self.latent_dim if self.dec_dim is None else self.dec_dim
        
        self.encoder = self._get_encoder(enc_dim, self.latent_dim)
        self.decoder = self._get_decoder(dec_dim, x.shape[-1])
        self.sigma_nn = self._get_sigma(self.latent_dim, self.latent_dim)
        
    def get_mu_sigma(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # code = self.encoder(x)
        # mu = code[:, :self.latent_dim]
        # sigma = torch.exp(code[:, self.latent_dim:])
        mu = self.encoder(x)
        sigma = self.sigma_nn(mu)
        return mu, sigma
        
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
                             device=self.device, dtype=torch.get_default_dtype())
        
        cost = torch.mean(torch.sum(torch.pow(x - recon, 2), dim=-1))
        
        kernel = self.kernel
        z_smp_diff_mtx = z_smp[None, ...] - z_smp[:, None, :]
        K_z_smp = kernel(z_smp_diff_mtx.flatten(end_dim=-2))
        z_diff_mtx = z_latent[None, ...] - z_latent[:, None, :]
        K_z = kernel(z_diff_mtx.flatten(end_dim=-2))
        z_dis_diff = z_smp[None, ...] - z_latent[:, None, :]
        K_dis = kernel(z_dis_diff.flatten(end_dim=-2))
        
        n = z_latent.shape[0]
        diag_mask = torch.eye(n, dtype=torch.bool, device=self.device).ravel()
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
            max = torch.amax(mse_diff, dim=-1)
        return torch.mean(max).item()
    
    # sample points from mean and std
    # if num is 1, mid dim is collapsed
    def gen_samples(self, mu: torch.Tensor, sigma: torch.Tensor, num: int = 1) -> torch.Tensor:
        epsilon = torch.normal(0, 1, (mu.shape[0], num, mu.shape[1]),
                               device=self.device, dtype=torch.get_default_dtype())
        if num == 1:
            epsilon = epsilon[:, 0, :]
        else:
            mu = mu[:, None, :]
            sigma = sigma[:, None, :]
        return mu + epsilon * sigma

    # get z = mu(x) + eps * sigma(x)
    def forward(self, x: torch.Tensor, samples_num: int = 1) -> torch.Tensor:
        mu_x, sigma_x = self.get_mu_sigma(x)
        z = self.gen_samples(mu_x, sigma_x, samples_num)
        return z
