import torch

class SFNet(torch.nn.Module):
    def _get_nn(self, dim_in: int, dim_out: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, dim_out),
            torch.nn.Sigmoid(),
        )
        
    def __init__(self, x_dim: int, times_n: int, device: torch.device) -> None:
        super().__init__()
        self.nn = self._get_nn(x_dim, times_n + 1).to(device)
        self.times_n = times_n
        
    def forward(self, x):
        steps_nn = self.nn(x) # (batch, t_n + 1)
        norm_sum = torch.sum(steps_nn, -1, keepdim=True)
        rev_csum = norm_sum - torch.cumsum(steps_nn, -1) # (batch, t_n + 1)
        surv_func = rev_csum / norm_sum # (batch, t_n + 1)
        surv_func = surv_func[:, :-1] # (batch, t_n)
        surv_steps = steps_nn[:, :-1] / norm_sum # (batch, t_n)
        return surv_func, surv_steps
        