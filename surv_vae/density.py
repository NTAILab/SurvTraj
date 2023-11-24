import torch
import numpy as np
from . import DEVICE
from sklearn.mixture._base import BaseMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import chi2

        
class MixturePDF(torch.nn.Module):
    
    DEFAULT_CLS_NUM = 10
    
    def __init__(self, cls_num: int = None, cls_threshold: float = None) -> None:
        super().__init__()
        self.sigma_inv = None
        self.det_sqrt = None
        self.mu = None
        self.dim = None
        self.c = None
        self.q = self.DEFAULT_CLS_NUM if cls_num is None else cls_num
        if cls_threshold is None:
            if cls_num <= 5:
                self.threshold = 0
            else:
                self.threshold = 1 / self.q
        else:
            self.threshold = cls_threshold        
    
    def _lazy_init(self, model: BaseMixture) -> None:
        if model.covariance_type != 'full':
            raise NotImplementedError()
        filter_idx = np.argwhere(model.weights_ > self.threshold).ravel()
        self.q = len(filter_idx)
        self.sigma_inv = torch.from_numpy(
            model.precisions_[filter_idx]).detach().to(DEVICE) # (q, m, m)
        chol = model.precisions_cholesky_
        det_sqrt = 1 / np.diagonal(chol[filter_idx], axis1=1, axis2=2).prod(-1)
        self.det_sqrt = torch.from_numpy(det_sqrt).detach()[None, :].to(DEVICE) # (1, q)
        self.mu = torch.from_numpy(
            model.means_[filter_idx]).detach()[None, ...].to(DEVICE) # (1, q, m)
        self.dim = self.mu.shape[-1]
        
        # weights = model.weights_[filter_idx]
        # weights = torch.from_numpy(weights / np.sum(weights)).to(DEVICE)[None, :]
        
        self.c = 1 / (np.power(2 * np.pi, self.dim / 2) * self.det_sqrt) # (1, q)
        
        # self.c = self.c * weights
        
    def conf_mask(self, x: torch.Tensor, q: float) -> torch.Tensor:
        x_shifted = x[:, None, :] - self.mu # (batch, q, m)
        x_prod = torch.sum(torch.sum(
                x_shifted[..., None, :] * self.sigma_inv[None, ...], dim=-1) * x_shifted, dim=-1)
        return x_prod <= chi2.ppf(q, self.dim)
        
    
    def fit(self, x: np.ndarray):
        mixture = BayesianGaussianMixture(n_components=self.q).fit(x)
        self._lazy_init(mixture)
        return self
    
    def forward(self, x: torch.Tensor):
        x_shifted = x[:, None, :] - self.mu # (batch, q, m)
        x_prod = -0.5 * \
            torch.sum(torch.sum(
                x_shifted[..., None, :] * self.sigma_inv[None, ...], dim=-1) * x_shifted, dim=-1)
        prob = self.c * torch.exp(x_prod) # (batch, q)
        return prob

    
    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).type(torch.get_default_dtype()).to(DEVICE)
            return self(x_t).cpu().numpy()
    
    def predict_conf_mask(self, x: np.ndarray, q: float) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).type(torch.get_default_dtype()).to(DEVICE)
            return self.conf_mask(x_t, q).cpu().numpy()
