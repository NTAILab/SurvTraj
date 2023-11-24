import torch
from . import DEVICE

class ConvStepSmoother(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.surv_conv_mask = torch.tensor([1, 2, 5, 2, 1], dtype=torch.get_default_dtype(), device=DEVICE)
        
    def forward(self, surv_steps: torch.Tensor, time_bg: torch.Tensor, time_in: torch.Tensor, single_time: bool) -> torch.Tensor:
        step_idx = torch.searchsorted(time_bg, time_in) # (t_n)
        idx_pattern = torch.arange(0, self.surv_conv_mask.shape[0], device=DEVICE) - self.surv_conv_mask.shape[0] // 2
        idx_pattern = idx_pattern[None, :] + step_idx[:, None] # (x_n, mask_len)
        clamp_mask = torch.logical_or(idx_pattern < 0, idx_pattern >= time_bg.shape[0])
        idx_pattern[clamp_mask] = 0
        if single_time:
            idxes = idx_pattern[:, None, :]
            steps_shape = (surv_steps.shape[0], surv_steps.shape[1], self.surv_conv_mask.shape[0])
        else:
            idxes = idx_pattern.ravel()[None, None, :]
            steps_shape = (surv_steps.shape[0], surv_steps.shape[1], time_in.shape[0], self.surv_conv_mask.shape[0])
        step = torch.take_along_dim(surv_steps, dim=-1, indices=idxes) # (x_n, z_n, t_n * mask_len)
        if not single_time:
            step = step.reshape(steps_shape)
        prob_mask = self.surv_conv_mask[None, :].broadcast_to((time_in.shape[0], self.surv_conv_mask.shape[0])).clone()
        prob_mask[clamp_mask] = 0
        prob_mask = prob_mask / torch.sum(prob_mask, dim=-1, keepdim=True)
        if single_time:
            prob_mask = prob_mask[:, None, :]
        else:
            prob_mask = prob_mask[None, None, ...]
        proba = torch.sum(step * prob_mask, dim=-1)
        return proba

class SoftminStepSmoother(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bandwidth = torch.tensor([1.0], dtype=torch.get_default_dtype(), device=DEVICE)
        self.bandwidth = torch.nn.Parameter(bandwidth, requires_grad=True)
        
    def forward(self, surv_steps: torch.Tensor, time_bg: torch.Tensor, time_in: torch.Tensor, single_time: bool) -> torch.Tensor:
        t_metric = torch.abs(time_bg[None, :] - time_in[:, None]) # (p_n, t_n) or (x_n, t_n)
        if single_time:
            t_metric = t_metric[:, None, :] # (x_n, 1, 1)
            surv_hist = surv_steps # (x_n, z_n, t_n)
        else:
            t_metric = t_metric[None, None, ...] # (1, 1, p_n, t_n)
            surv_hist = surv_steps[..., None, :] # (x_n, z_n, 1, t_n)
        bandwidth = self.bandwidth.clamp_min(1e-6)
        t_weights = torch.nn.functional.softmin(bandwidth * t_metric, dim=-1)
        # total_weights = t_weights * surv_hist
        # total_weights = total_weights / torch.sum(total_weights, dim=-1, keepdim=True)
        proba = torch.sum(t_weights * surv_hist, dim=-1)
        return proba