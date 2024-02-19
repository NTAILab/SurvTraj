import numpy as np
import sksurv.datasets as ds
from sksurv.column import encode_categorical, standardize, categorical_to_numeric
from typing import Set, TypeVar, Tuple, Dict, List, Optional
import torch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from .model import SurvTraj
from pandas.api.types import is_categorical_dtype

TYPE = [('censor', '?'), ('time', 'f8')]

T = TypeVar('T')

def get_all_subclasses(cls: T) -> Set[T]:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])

class sksurv_loader():
    def __init__(self, z_normalization=True) -> None:
        self.z_norm = z_normalization
    
    def __getattr__(self, key):
        X, y = getattr(ds, key)()
        if self.z_norm:
            X = standardize(X)
        # X = encode_categorical(X)
        X = categorical_to_numeric(X)
        return X.to_numpy(), y.astype(TYPE)

    def get_feat_info(self, ds_name):
        X, y = getattr(ds, ds_name)()
        
        is_cat = lambda series: is_categorical_dtype(series.dtype) or series.dtype == "O"
        
        feat_names = []
        cat_dict = dict()
        cont_dict = dict()
        for name, s in X.items():
            feat_names.append(name)
            if is_cat(s):
                # cat_set.add(name)
                cat_dict[name] = list(s.dtype.categories)
            else:
                cont_dict[name] = {
                    'mean': X[name].mean(),
                    'std': X[name].std(),
                }
        return feat_names, cont_dict, cat_dict
        
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

def get_traject_plot(x: np.ndarray, t: np.ndarray, 
                     plot_kw: Optional[Dict] = None, 
                     labels: Optional[List[str]] = None) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(1, 1, dpi=100, figsize=(6, 6))
    if plot_kw is None:
        plot_kw = dict()
    for i in range(x.shape[1]):
        lbl = f'$x_{{{i + 1}}}$' if labels is None else labels[i]
        ax.plot(t, x[:, i], label=lbl, **plot_kw)
    ax.grid()
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    return fig, ax

def get_traj_plot_ci(model: SurvTraj, x: np.ndarray, t: np.ndarray, 
                      traj_num: int = 100, conf_p: float = 0.25,
                      labels: Optional[List[str]] = None,
                      cat_names: Optional[Dict[str, List[str]]] = None,
                      post_processor = None) -> Tuple[Figure, List[Axes]]:
    assert x.ndim == t.ndim == 1
    assert conf_p < 0.5
    if post_processor is None:
        post_processor = lambda x: x
    cat_n = 0 if cat_names is None else len(cat_names)
    cont_n = x.shape[0] - cat_n
    x_in = np.tile(x[None, :], (traj_num, 1))
    t_in = np.tile(t[None, :], (traj_num, 1))
    x_traj = model.predict_trajectory(x_in, t_in)
    x_traj = post_processor(x_traj)
    low_quantile = np.quantile(x_traj, conf_p, 0)
    up_quantile = np.quantile(x_traj, 1 - conf_p, 0)
    median = np.median(x_traj, 0)
    fig, cont_axes = plt.subplots(cont_n, 1, dpi=100,
                             figsize=(6, 2 * cont_n), 
                             layout='tight')
    fig.suptitle('Continuous features')
    if cat_n > 0:
        fig, cat_axes = plt.subplots(cat_n, 1, dpi=100,
                                figsize=(6, 2 * cat_n), 
                                layout='tight')
        fig.suptitle('Categorical features')
    i_cont = 0
    i_cat = 0
    for i in range(x.shape[0]):
        lbl = f'$x_{{{i + 1}}}$' if labels is None else labels[i]
        if cat_n > 0 and lbl in cat_names:
            axes = cat_axes
            j = i_cat
            axes[j].set_yticks(np.arange(len(cat_names[lbl])), cat_names[lbl])
            i_cat += 1
        else:
            axes = cont_axes
            j = i_cont
            axes[j].set_ylabel('x')
            i_cont += 1
        lq = low_quantile[:, i]
        uq = up_quantile[:, i]
        med = median[:, i]
        axes[j].fill_between(t, lq, uq, alpha=0.25)
        axes[j].plot(t, med, 's--')
        axes[j].set_title(lbl)
        axes[j].set_xlabel('t')
        axes[j].grid()
    return fig, cont_axes
    