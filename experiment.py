from argparse import ArgumentParser
from abc import ABC, abstractstaticmethod, abstractmethod
from time import gmtime, strftime, time
import numpy as np
from surv_vae.surv_mixup import SurvivalMixup
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from surv_vae.utility import TYPE, sksurv_loader
from sksurv.ensemble import RandomSurvivalForest
from typing import Optional, Tuple, Dict, Type
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

class Dataset(ABC):
    # full data is N samples
    # then test is test_part * N,
    # train is (1 - test_part) * (1 - val_part) * N
    # validation is (1 - test_part) * val_part * N
    # stratification is done according to the delta
    def __init__(self, x: np.ndarray, y: np.recarray,
                 val_part: float, test_part: float,
                 seed: Optional[int] = None):
        self.x = x
        self.y = y
        self.val_part = val_part
        self.test_part = test_part
        self.seed = seed if seed is not None else np.random.randint(0, 4096)
        self.dim = x.shape[1]
        
    def get_train_test_set(self) -> Tuple[np.ndarray, np.recarray, np.ndarray, np.recarray]:
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, 
                                                            test_size=self.test_part,\
                                                            random_state=self.seed,
                                                            stratify=self.y[TYPE[0][0]])
        return x_train, y_train, x_test, y_test
    
    def get_ttv_set(self) -> Tuple[np.ndarray, np.recarray, np.ndarray, np.recarray, np.ndarray, np.recarray]:
        tt_set = self.get_train_test_set()
        x_train, x_val, y_train, y_val = train_test_split(tt_set[0], tt_set[1],
                                                          test_size=self.val_part,
                                                          random_state=self.seed,
                                                          stratify=tt_set[1][TYPE[0][0]])
        return x_train, y_train, tt_set[2], tt_set[3], x_val, y_val
    
    @abstractstaticmethod
    def get_name() -> str:
        pass

class Veterans(Dataset):
    def __init__(self, val_part: float, test_part: float, seed: int | None = None):
        loader = sksurv_loader()
        x, y = loader.load_veterans_lung_cancer
        super().__init__(x, y, val_part, test_part, seed)
    
    @staticmethod
    def get_name() -> str:
        return 'veterans'
    
class WHAS500(Dataset):
    def __init__(self, val_part: float, test_part: float, seed: int | None = None):
        loader = sksurv_loader()
        x, y = loader.load_whas500
        super().__init__(x, y, val_part, test_part, seed)
    
    @staticmethod
    def get_name() -> str:
        return 'whas500'
    
class GBSG2(Dataset):
    def __init__(self, val_part: float, test_part: float, seed: int | None = None):
        loader = sksurv_loader()
        x, y = loader.load_gbsg2
        super().__init__(x, y, val_part, test_part, seed)
    
    @staticmethod
    def get_name() -> str:
        return 'gbsg2'
    
def get_ds_map() -> Dict[str, Dataset]:
    ds_list = Dataset.__subclasses__()
    ds_map = dict()
    for ds_cls in ds_list:
        ds_map[ds_cls.get_name()] = ds_cls
    return ds_map

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
    
class SurvMixupWrapper(ModelWrapper):
    def __init__(self, samples_num: int,
                 latent_dim: int, 
                 regular_coef: float, 
                 sigma_z: float = 1,
                 batch_num: int = 20,
                 epochs: int = 100,
                 lr_rate: float = 0.001,
                 benk_vae_loss_rat: float = 0.2,
                 c_ind_temp: float = 1.0,
                 gumbel_tau: float = 1.0,
                 train_bg_part: float = 0.6,
                 cens_cls_model = None,
                 batch_load: Optional[int] = 256,
                 patience: int = 10):
        vae_kw = {
            'latent_dim': latent_dim,
            'regular_coef': regular_coef,
            'sigma_z': sigma_z,
        }
        if cens_cls_model is None:
            cens_cls_model = RandomForestClassifier()
        self.model = SurvivalMixup(vae_kw, samples_num, batch_num,
                                   epochs, lr_rate, benk_vae_loss_rat,
                                   c_ind_temp, gumbel_tau, train_bg_part,
                                   cens_cls_model, batch_load, patience)
    
    def fit(self, ds: Dataset):
        x_nn_train, y_nn_train, _, _, *val_set = ds.get_ttv_set()
        self.model.fit(x_nn_train, y_nn_train, val_set)
        
    def score(self, ds: Dataset):
        _, _, x_test, y_test = ds.get_train_test_set()
        return self.model.score(x_test, y_test)
    
    @staticmethod
    def get_name() -> str:
        return 'Surv VAE'
    
class SurvForestWrapper(ModelWrapper):
    def __init__(self, folds_n: int, cv_iters: int, 
                 n_jobs: int, seed: Optional[int] = None):
        self.folds_n = folds_n
        self.forest_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'min_samples_leaf': [1, 0.01, 0.05, 0.1],
        }
        self.cv_iters = cv_iters
        self.seed = seed
        self.n_jobs = n_jobs
        self.model = None
    
    def fit(self, ds: Dataset):
        cv_strat = KFold(self.folds_n, random_state=self.seed, shuffle=True) # todo: make out how to use StratifiedKFold
        cv = RandomizedSearchCV(RandomSurvivalForest(), self.forest_grid, cv=cv_strat, 
                                n_iter=self.cv_iters, n_jobs=self.n_jobs,
                                random_state=self.seed)
        x_train, y_train, _, _ = ds.get_train_test_set()
        cv.fit(x_train, y_train)
        self.model = cv.best_estimator_
    
    def score(self, ds: Dataset):
        _, _, x_test, y_test = ds.get_train_test_set()
        return self.model.score(x_test, y_test)
    
    @staticmethod
    def get_name() -> str:
        return 'Surv RForest'

class Experiment():
    METRIC_NAME = 'C_index'
    
    def __init__(self, dataset: Dataset, 
                 iter_n: int, folds_n: int, cv_iters: int, n_jobs: int, 
                 seed:Optional[int]=None, output_file: Optional[str]=None):
        self.ds = dataset
        self.iter_n = iter_n
        self.out_file = output_file
        self.rng = np.random.default_rng(seed)
        self.folds_n = folds_n
        self.cv_iters = cv_iters
        self.n_jobs = n_jobs
        self.models_params = self.get_all_models_params()
        self.exp_results = None
        
    def get_all_models_params(self):
        params = dict()
        models_dict = self.get_models()
        for model_name, model_cls in models_dict.items():
            if model_cls is SurvMixupWrapper:
                params[model_name] = lambda : {
                    'samples_num': 48,
                    'latent_dim': int(self.ds.dim * 1.5), 
                    'regular_coef': 66, 
                    'sigma_z': 1,
                    'batch_num': 20,
                    'epochs': 100,
                    'lr_rate': 0.001,
                    'benk_vae_loss_rat': 0.2,
                    'c_ind_temp': 1.0,
                    'gumbel_tau': 1.0,
                    'train_bg_part': 0.6,
                    'cens_cls_model': RandomForestClassifier(),
                    'batch_load': 256,
                    'patience': 5
                }
            else:
                params[model_name] = lambda : {
                    'folds_n': self.folds_n, 
                    'cv_iters': self.cv_iters, 
                    'n_jobs': self.n_jobs, 
                    'seed': self.rng.integers(1, 4096)
                }
        return params

    def get_model_params(self, model_name: str):
        return self.models_params[model_name]()
        
    def get_models(self) -> Dict[str, Type[ModelWrapper]]:
        models = ModelWrapper.__subclasses__()
        result = dict((model.get_name(), model) for model in models)
        return result
    
    def exp_iter(self):
        iter_results = dict()
        models = self.get_models()
        for model_name, model_cls in models.items():
            model_params = self.get_model_params(model_name)
            model = model_cls(**model_params)
            model.fit(self.ds)
            iter_results[f"{model_name} {self.METRIC_NAME}"] = model.score(self.ds)
        return iter_results
            
    
    def run(self):
        self.exp_results = defaultdict(list)
        for i in range(self.iter_n):
            self.ds.seed = self.rng.integers(1, 4096)
            iter_results = self.exp_iter()
            for key, val in iter_results.items():
                self.exp_results[key].append(val)
            print(f'Experiment iteration {i} is passed')
        return self.exp_results
    
    def dump_results(self, output_file: Optional[str]=None):
        if output_file is None:
            output_file = self.out_file
        with open(output_file, 'w') as file:
            json.dump(self.exp_results, file, indent=1)

DEFAULT_DS = Veterans.get_name()
DEFAULT_EXP_ITERS = 3
DEFAULT_VAL_PART = 0.25
DEFAULT_TEST_PART = 0.4
DEFAULT_FOLDS_N = 3
DEFAULT_CV_JOBS = 6
DEFAULT_CV_ITERS = 10

def cli():
    parser = ArgumentParser('surv_vae experiment script')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DS, help='Dataset name')
    parser.add_argument('--iter_n', type=int, default=DEFAULT_EXP_ITERS,
                        help='Number of the iterations')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for the experiment')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--test_part', type=float, default=DEFAULT_TEST_PART, help='Partition for the test data')
    parser.add_argument('--val_part', type=float, default=DEFAULT_VAL_PART, help='Partition for the validation set')
    parser.add_argument('--folds_n', type=int, default=DEFAULT_FOLDS_N, help='Number of the folds in the CV')
    parser.add_argument('--cv_jobs', type=int, default=DEFAULT_CV_JOBS, help='Number of the parallel jobs in the CV')
    parser.add_argument('--cv_iters', type=int, default=DEFAULT_CV_ITERS, help='Number of the iterations in the randomized CV')
    

    args = parser.parse_args()
    
    assert args.iter_n > 0, 'Iterations number must be positive'
    assert args.folds_n > 0, 'Folds number must be positive'
    assert args.cv_jobs > 0, 'CV jobs number must be positive'
    assert args.cv_iters > 0, 'CV iterations number must be positive'
    assert args.seed is None or args.seed >= 0, 'Random seed must be non-negative'
    assert 0 < args.test_part < 1, 'Test part must be in (0, 1)'
    assert 0 < args.val_part < 1, 'Test part must be in (0, 1)'
    
    if args.output is None:
        output_file = f"{args.dataset}_exp {strftime('%m_%d %H_%M_%S', gmtime())}.json"
    else:
        output_file = args.output
    
    ds_map = get_ds_map()
    ds_cls = ds_map.get(args.dataset, None)
    if ds_cls is None:
        raise AttributeError(f"Dataset {args.dataset} is not found")
    ds = ds_cls(args.val_part, args.test_part, args.seed)
    return Experiment(ds, args.iter_n, args.folds_n, args.cv_iters, args.cv_jobs, args.seed, output_file)
    

if __name__ == '__main__':
    experiment = cli()
    time_stamp = time()
    try:
        experiment.run()
        experiment.dump_results()
    except Exception as e:
        backup_name = f'experiment_{experiment.ds.get_name()}_backup_' + strftime('%m_%d %H_%M_%S', gmtime()) + '.json'
        with open(backup_name, 'w') as file:
            json.dump(experiment.exp_results, file, indent=1)
        raise e
    time_elapsed = int(time() - time_stamp)
    print('Time elapsed: ', time_elapsed // 60, ' min.', time_elapsed % 60, ' sec.')
    