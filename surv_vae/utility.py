import numpy as np
import sksurv.datasets as ds
from sksurv.column import encode_categorical, standardize
import torch

TYPE = [('censor', '?'), ('time', 'f8')]

class sksurv_loader():
    def __init__(self, z_normalization=True) -> None:
        self.z_norm = z_normalization
    
    def __getattr__(self, key):
        X, y = getattr(ds, key)()
        X = encode_categorical(X)
        if self.z_norm:
            X = standardize(X)
        return X.to_numpy(), y.astype(TYPE)

# separate arrays to the structured one
# first field - censoring flag (bool)
# second field - time to event
def get_str_array(T: np.ndarray, D: np.ndarray) -> np.recarray:
    assert T.shape[0] == D.shape[0]
    str_array = np.ndarray(shape=(T.shape[0]), dtype=TYPE)
    str_array['censor'] = D
    str_array['time'] = T
    return str_array


class TempTanh(torch.nn.Module):
    def __init__(self, gain: float = 1):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * torch.tanh(x)

