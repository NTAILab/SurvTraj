from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from numpy.random import RandomState
from surv_traj.model import SurvTraj
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from typing import Any, Iterator, Optional
from abc import ABC, abstractmethod, abstractstaticmethod
from datasets import Dataset
import numpy as np
from surv_traj.utility import TYPE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from numba import njit
import warnings


class ModelWrapper(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, ds: Dataset) -> None:
        pass
    
    @abstractstaticmethod
    def get_name() -> str:
        pass
    
    @abstractmethod
    def score(self, ds: Dataset) -> float:
        pass
    
class SurvTrajWrapper(ModelWrapper):
    def __init__(self, samples_num: int,
                 latent_dim: int, 
                 regular_coef: float, 
                 sigma_z: float = 1,
                 batch_num: int = 20,
                 epochs: int = 100,
                 lr_rate: float = 0.001,
                 c_ind_weight: float = 1.0,
                 vae_weight: float = 1.0,
                 traj_weight: float = 1.0,
                 likelihood_weight: float = 1.0,
                 c_ind_temp: float = 1.0,
                 gumbel_tau: float = 1.0,
                 train_bg_part: float = 0.6,
                 traj_penalty_points: int = 10,
                 cens_cls_model = None,
                 batch_load: Optional[int] = 256,
                 patience: int = 10,
                 device: str = 'cpu'):
        vae_kw = {
            'latent_dim': latent_dim,
            'regular_coef': regular_coef,
            'sigma_z': sigma_z,
        }
        if cens_cls_model is None:
            cens_cls_model = RandomForestClassifier()
        self.model = SurvTraj(vae_kw, samples_num, batch_num,
                                   epochs, lr_rate, c_ind_weight,
                                   vae_weight, traj_weight, likelihood_weight,
                                   c_ind_temp, gumbel_tau, train_bg_part, traj_penalty_points,
                                   cens_cls_model, batch_load, patience, device)
        super().__init__()
    
    def fit(self, ds: Dataset):
        x_nn_train, y_nn_train, x_test, y_test, *val_set = ds.get_ttv_set()
        self.model.fit(x_nn_train, y_nn_train, val_set)
        
    def score(self, ds: Dataset):
        _, _, x_test, y_test = ds.get_train_test_set()
        return self.model.score(x_test, y_test)
    
    @staticmethod
    def get_name() -> str:
        return 'Surv VAE'
    
class DeltaStratifiedKFold(StratifiedKFold):
    def __init__(self, n_splits: int = 5, *, 
                 shuffle: bool = False, 
                 random_state: int | RandomState | None = None) -> None:
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
    
    def split(self, X: np.ndarray, 
              y: np.recarray, 
              groups: Any = None) -> Iterator[Any]:
        delta = y[TYPE[0][0]]
        return super().split(X, delta, groups)
    
class CVWrapper(ModelWrapper):
    def __init__(self, folds_n: int, cv_iters: int, 
                 n_jobs: int, seed: Optional[int] = None):
        self.cv_iters = cv_iters
        self.seed = seed
        self.n_jobs = n_jobs
        self.folds_n = folds_n
        self.model = None
    
    @abstractmethod
    def get_params_grid(self):
        pass
    
    @abstractmethod
    def get_base_model(self):
        pass
    
    def fit(self, ds: Dataset):
        cv_strat = DeltaStratifiedKFold(self.folds_n, 
                                        random_state=self.seed, 
                                        shuffle=True)
        cv = RandomizedSearchCV(self.get_base_model(), self.get_params_grid(), cv=cv_strat, 
                                n_iter=self.cv_iters, n_jobs=self.n_jobs,
                                random_state=self.seed)
        x_train, y_train, _, _ = ds.get_train_test_set()
        cv.fit(x_train, y_train)
        self.model = cv.best_estimator_
    
    def score(self, ds: Dataset):
        _, _, x_test, y_test = ds.get_train_test_set()
        return self.model.score(x_test, y_test)
    
class SurvForestWrapper(CVWrapper):
    def __init__(self, folds_n: int, cv_iters: int, 
                 n_jobs: int, seed: Optional[int] = None):
        super().__init__(folds_n, cv_iters, n_jobs, seed)
    
    def get_params_grid(self):
        return {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'min_samples_leaf': [1, 0.01, 0.05, 0.1],
        }
        
    def get_base_model(self):
        return RandomSurvivalForest(random_state=self.seed)
    
    @staticmethod
    def get_name() -> str:
        return 'Surv RForest'

@njit
def nw_helper_func(W, D):
    k = W.shape[0]
    n = W.shape[1]
    S = np.empty((k, n), np.float32)
    if D[0] == 1:
        S[:, 0] = 1 - W[:, 0]
    else:
        S[:, 0] = 1
    for i in range(1, n):
        if D[i] == 0:
            S[:, i] = 1 * S[:, i - 1]
            continue
        weight_sum = np.zeros(k, dtype=np.float32)
        for j in range(i):
            weight_sum += W[:, j]
        cur_S = 1 - W[:, i] / (1 - weight_sum)
        for j in range(k):
            if not np.isfinite(cur_S[j]) or cur_S[j] < 0:
                cur_S[j] = 1
        S[:, i] = S[:, i - 1] * cur_S
    return S

def sf_to_t(S, T):
    t_diff = T[1:] - T[:-1]
    integral = S[:, :-1] * t_diff[None, :]
    return T[0] + np.sum(integral, axis=-1)
    
class Beran():
    # random state is only for the interface consistency
    def __init__(self, gamma=None, random_state=None):
        self.gamma = 1 if gamma is None else gamma

    def predict(self, x):
        np.seterr(invalid='ignore', divide='ignore')
        W = rbf_kernel(x, self.x_train, gamma=self.gamma)
        W = W / np.sum(W, axis=-1, keepdims=True)
        W[np.any(np.logical_not(np.isfinite(W)), axis=-1), :] = 1 / self.x_train.shape[0]
        S = nw_helper_func(W, self.delta)
        if np.any(np.logical_not(np.isfinite(S))):
            raise ValueError('nan or inf in S')
        return sf_to_t(S, self.T)

    def fit(self, x: np.ndarray, y: np.recarray):
        self.T = y['time']
        sort_args = np.argsort(self.T)
        self.T = self.T[sort_args]
        self.x_train = x[sort_args]
        self.delta = y['censor'].astype(np.int64)[sort_args]
        return self

    def get_params(self, deep: bool=False):
        return {'gamma': self.gamma}

    def set_params(self, gamma: float):
        self.gamma = gamma
        return self
    
    def score(self, x: np.ndarray, y: np.recarray) -> float:
        T = self.predict(x)
        c_ind, *_ = concordance_index_censored(y['censor'], y['time'], -T)
        return c_ind

    
class BeranWrapper(CVWrapper):
    def __init__(self, folds_n: int, cv_iters: int, 
                 n_jobs: int, seed: Optional[int] = None):
        super().__init__(folds_n, cv_iters, n_jobs, seed)
        
    def get_params_grid(self):
        return  {
            'gamma': [10 ** i for i in range(-3, 4)] + [0.5, 5, 50, 200, 500, 700]
        }
        
    def get_base_model(self):
        return Beran()
        
    @staticmethod
    def get_name() -> str:
        return 'Beran'


class CoxnetWrapper(CVWrapper):
    def __init__(self, folds_n: int, 
                 cv_iters: int, n_jobs: int, 
                 seed: int | None = None):
        super().__init__(folds_n, cv_iters, n_jobs, seed)
        self.alphas = None
        self.n_alphas = 30
    
    def get_params_grid(self):
        return {
            'alphas': [[a] for a in self.alphas],
        }
    
    def fit(self, ds: Dataset):
        x_train, y_train, _, _ = ds.get_train_test_set()
        cox = CoxnetSurvivalAnalysis(n_alphas=self.n_alphas).fit(x_train, y_train)
        self.alphas = [a for a in cox.alphas_]
        return super().fit(ds)
    
    def get_base_model(self):
        return CoxnetSurvivalAnalysis(n_alphas=1, l1_ratio=0.75, normalize=False)
    
    @staticmethod
    def get_name() -> str:
        return 'Coxnet'
