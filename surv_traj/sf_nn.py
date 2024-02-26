import torch

class SFNet(torch.nn.Module):
    def _get_nn(self, dim_in: int, dim_out: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, dim_out),
        )
        
    def __init__(self, x_dim: int, times_n: int, device: torch.device) -> None:
        super().__init__()
        self.nn = self._get_nn(x_dim, times_n + 1).to(device)
        self.times_n = times_n
        
    def forward(self, x):
        pi_logit = self.nn(x) # (batch, t_n + 1)
        steps_nn = torch.nn.functional.softmax(pi_logit, -1) # (batch, t_n + 1)
        # norm_sum = torch.sum(steps_nn, -1, keepdim=True)
        rev_csum = 1 - torch.cumsum(steps_nn, -1) # (batch, t_n + 1)
        # surv_func = rev_csum / norm_sum # (batch, t_n + 1)
        surv_func = rev_csum[..., :-1] # (batch, t_n)
        surv_logits = steps_nn[..., :-1]  # (batch, t_n)
        return surv_func, surv_logits
        