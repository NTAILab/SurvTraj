import torch
from numpy import sqrt
from . import EPSILON, DEVICE

class NNKernel(torch.nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        # s = int(sqrt(m))
        self.sparse = torch.nn.Sequential(
            torch.nn.Linear(m, 2 * m),
            torch.nn.ReLU6()
        ).to(DEVICE)
        self.sequent = torch.nn.Sequential(
            torch.nn.Linear(2 * m, m),
            torch.nn.ReLU6(),
            # torch.nn.Linear(m, m),
            # torch.nn.Tanh(),
            torch.nn.Linear(m, 1),
            torch.nn.Softplus()
        ).to(DEVICE)

    def forward(self, x_1, x_2):
        sparse_1 = self.sparse(x_1)
        sparse_2 = self.sparse(x_2)
        total_in = torch.abs(sparse_1 - sparse_2)
        W = self.sequent(total_in)
        sum = torch.sum(W, dim=-2, keepdim=True).broadcast_to(W.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        norm_weights = W / sum
        norm_weights[bad_idx] = 0
        return norm_weights
    
class SimpleGauss(torch.nn.Module):
    def __init__(self, norm_axis: int=-2):
        super().__init__()
        self.bandwidth = torch.nn.parameter.Parameter(
                            torch.tensor([1.0],
                            dtype=torch.get_default_dtype(), device=DEVICE),
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
        

class BENK(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.kernel = SimpleGauss()
        # self.kernel = NNKernel(dim)

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
        # surv_steps = filtered_xi * surv_func
        sf_alias = surv_func # torch.concat((torch.ones((surv_func.shape[0], 1), device=DEVICE), surv_func), dim=-1)
        surv_steps = sf_alias[:, :-1] - sf_alias[:, 1:]
        return surv_func, surv_steps

    def predict(self, *args):
        self.eval()
        with torch.no_grad():
            return self(*args).cpu().numpy()
