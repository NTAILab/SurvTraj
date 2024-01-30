import torch

class SimpleGauss(torch.nn.Module):
    def __init__(self, device: torch.device, norm_axis: int=-2):
        super().__init__()
        self.bandwidth = torch.nn.parameter.Parameter(
                            torch.tensor([1.0],
                            dtype=torch.get_default_dtype(), device=device),
                            requires_grad=True)
        self.dim = norm_axis
    
    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        metric = torch.linalg.vector_norm(x_1 - x_2, dim=-1, keepdim=True)
        bandwidth = torch.clamp(self.bandwidth, min=0.1, max=10)
        weights = torch.exp(-metric / bandwidth)
        sum = torch.sum(weights, dim=self.dim, keepdim=True).broadcast_to(weights.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        norm_weights = weights / sum
        norm_weights[bad_idx] = 0
        return norm_weights # (x_n_1, x_n_2, ..., 1)
        
class Beran(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.kernel = SimpleGauss(device)

    def forward(self, x_in, delta_in, x_p):
        n = x_in.shape[1]
        x_p_repeat = x_p[:, None, :].expand(-1, n, -1)
        W = self.kernel(x_in, x_p_repeat)[..., 0] # (batch, n)
        w_cumsum = torch.cumsum(W, dim=1)
        shifted_w_cumsum = w_cumsum - W
        ones = torch.ones_like(shifted_w_cumsum)
        bad_idx = torch.isclose(shifted_w_cumsum, ones) | torch.isclose(w_cumsum, ones)
        shifted_w_cumsum[bad_idx] = 0.0
        w_cumsum[bad_idx] = 0.0

        xi = torch.log(1.0 - shifted_w_cumsum)
        xi -= torch.log(1.0 - w_cumsum)

        filtered_xi = delta_in * xi
        hazards = torch.cumsum(filtered_xi, dim=1)
        surv_func = torch.exp(-hazards) 
        surv_steps = surv_func[:, :-1] - surv_func[:, 1:]
        return surv_func, surv_steps
