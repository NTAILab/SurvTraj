import torch
import numpy as np
from typing import Optional
from . import DEVICE, EPSILON
from .smoother import ConvStepSmoother


class NNKaplanMeier(torch.nn.Module):
    def _get_internal_nn(self, dim: int):
        return torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(dim, dim),
                                   torch.nn.Sigmoid()).to(DEVICE)

    def __init__(self) -> None:
        super().__init__()
        self.T = None
        self.nn = None
        # self.smoother = ConvStepSmoother()

    def _lazy_init(self):
        self.nn = self._get_internal_nn(self.T.shape[0])

    def fit(self, y: np.recarray) -> 'NNKaplanMeier':
        T, delta = y['time'], y['censor']
        T = np.unique(np.concatenate(([0], T[delta])))
        self.T = torch.from_numpy(T).type(torch.get_default_dtype()).to(DEVICE)
        self._lazy_init()
        return self

    def forward(self, t: torch.Tensor):
        S = self.nn(self.T[None, :])[0, :]
        # S = torch.flip(S, (-1,))
        cum_sum = torch.cumsum(S, -1)
        S = S + cum_sum[-1] - cum_sum
        # S = torch.flip(S, (-1,))
        S = S / S[0]
        # S = torch.clamp_min(S, EPSILON)
        surv_steps = S[:-1] - S[1:]
        proba = self.smoother(surv_steps[None, None, :], self.T[1:], t, False)
        proba = proba[0, 0, :].clamp_max_(EPSILON)
        return proba


class PiHead(torch.nn.Module):
    def __init__(self, y: Optional[np.recarray] = None) -> None:
        super().__init__()
        self.km = NNKaplanMeier().fit(y) if y is not None else None

    def forward(self, surv_steps: torch.Tensor, pdf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.km is not None:       
            km_steps = self.km(t.ravel())
            if surv_steps.ndim == 2:
                km_steps = km_steps[:, None].broadcast_to(surv_steps.shape)
            else:
                km_steps = km_steps[None, None, :].broadcast_to(surv_steps.shape)
        
        # sum = torch.sum(surv_steps, dim=1, keepdim=True).broadcast_to(surv_steps.shape).clone()
        # bad_idx = sum < 1e-13
        # sum[bad_idx] = 1
        # surv_steps = surv_steps / sum
        # surv_steps[bad_idx] = 0
        
        # sum = torch.sum(pdf, dim=1, keepdim=True).broadcast_to(pdf.shape).clone()
        # bad_idx = sum < 1e-13
        # sum[bad_idx] = 1
        # pdf = pdf / sum  # (x_n, z_n)
        # pdf[bad_idx] = 0
        
        pi = surv_steps * pdf # (x_n, z_n, p_n)
        if self.km is not None:
            pi = pi / km_steps
        sum = torch.sum(pi, dim=1, keepdim=True).broadcast_to(pi.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        pi = pi / sum  # (x_n, z_n)
        pi[bad_idx] = 0
        return pi
