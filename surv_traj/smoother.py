import torch

class DummySmoother(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, surv_steps: torch.Tensor, time_bg: torch.Tensor, time_in: torch.Tensor, z_smp_n: int):
        time_idx = torch.searchsorted(time_bg, time_in.ravel()).clamp_max_(time_bg.shape[0] - 1) # (batch * p_n)
        if time_in.ndim == 2:
            time_idx = torch.reshape(time_idx, time_in.shape)
        else:
            time_idx = time_idx[:, None]
        # time_idx: (batch, p_n)
        surv_steps = torch.reshape(surv_steps, (time_in.shape[0], z_smp_n, -1, time_bg.shape[0])) # (batch, z_n, p_n, t_n)
        steps = torch.take_along_dim(surv_steps, time_idx[:, None, :, None], dim=-1)[..., 0] # (batch, z_n, p_n)
        if steps.shape[-1] == 1:
            steps.squeeze_(-1)
        return steps

class SoftminStepSmoother(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        bandwidth = torch.tensor([1.0], dtype=torch.get_default_dtype(), device=device)
        self.bandwidth = torch.nn.Parameter(bandwidth, requires_grad=True)
        
    def forward(self, surv_steps: torch.Tensor, time_bg: torch.Tensor, time_in: torch.Tensor, z_smp_n: int) -> torch.Tensor:
        # surv_steps: (batch * z_i_n, t_n), z_i_n = samples * p_n or samples
        # time_in: (batch) or (batch, p_n)
        # time_bg: (t_n)
        if time_in.ndim == 1:
            time_in = time_in[:, None]
        surv_steps = torch.reshape(surv_steps, (time_in.shape[0], z_smp_n, -1, time_bg.shape[0])) # (batch, z_n, p_n, t_n)
        t_metric = torch.abs(time_bg[None, None, :] - time_in[..., None]) # (batch, p_n, t_n)
        bandwidth = self.bandwidth.clamp_min(1e-6)
        t_weights = torch.nn.functional.softmin(bandwidth * t_metric, dim=-1) # (batch, p_n, t_n)
        proba = torch.sum(t_weights[:, None, ...] * surv_steps, dim=-1) # (batch, z_n, p_n)
        if proba.shape[-1] == 1:
            proba.squeeze_(-1) # (batch, z_n)
        # assert not torch.any(torch.isnan(proba))
        return proba
