import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
EPSILON = 1e-8
