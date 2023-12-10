from argparse import ArgumentParser
from time import gmtime, strftime, time
import numpy as np
from datasets import Dataset, get_ds_map, Veterans
from sklearn.ensemble import RandomForestClassifier
from models import ModelWrapper, SurvMixupWrapper
from typing import Optional, Dict, Type
from collections import defaultdict
from inspect import isabstract
from surv_vae.utility import get_all_subclasses
import json

class Experiment():
    METRIC_NAME = 'C_index'
    
    def __init__(self, dataset: Dataset, iter_n: int,
                 epochs: int, patience: int,
                 folds_n: int, cv_iters: int, n_jobs: int, 
                 seed:Optional[int]=None, output_file: Optional[str]=None):
        self.ds = dataset
        self.iter_n = iter_n
        self.epochs = epochs
        self.patience = patience
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
                    'latent_dim': int(np.ceil(self.ds.dim * 1.8)), 
                    'regular_coef': 40, 
                    'sigma_z': 1,
                    'batch_num': 16,
                    'epochs': self.epochs,
                    'lr_rate': 2e-3,
                    'benk_vae_loss_rat': 0.5,
                    'c_ind_temp': 1.0,
                    'gumbel_tau': 0.75,
                    'train_bg_part': 0.6,
                    'cens_cls_model': None,
                    'batch_load': 256,
                    'patience': self.patience
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
        models = get_all_subclasses(ModelWrapper)
        result = dict((model.get_name(), model) for model in models if not isabstract(model))
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
            
    
    def run(self, verbose: int = 0):
        self.exp_results = defaultdict(list)
        verb_counter = 0
        for i in range(self.iter_n):
            self.ds.seed = self.rng.integers(1, 4096)
            iter_results = self.exp_iter()
            for key, val in iter_results.items():
                self.exp_results[key].append(val)
            if verbose > 0:
                verb_counter += 1
                if verb_counter == verbose:
                    verb_counter = 0
                    self.dump_results()
            print(f'Experiment iteration {i} is passed')
        return self.exp_results
    
    def dump_results(self, output_file: Optional[str]=None):
        if output_file is None:
            output_file = self.out_file
        with open(output_file, 'w') as file:
            json.dump(self.exp_results, file, indent=1)

DEFAULT_DS = Veterans.get_name()
DEFAULT_EXP_ITERS = 10
DEFAULT_VAL_PART = 0.33
DEFAULT_TEST_PART = 0.4
DEFAULT_FOLDS_N = 3
DEFAULT_CV_JOBS = 4
DEFAULT_CV_ITERS = 10
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 20

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
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs during nn training')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, help='Number of validation patience epochs during nn training')

    args = parser.parse_args()
    
    assert args.iter_n > 0, 'Iterations number must be positive'
    assert args.folds_n > 0, 'Folds number must be positive'
    assert args.cv_jobs > 0, 'CV jobs number must be positive'
    assert args.cv_iters > 0, 'CV iterations number must be positive'
    assert args.epochs > 0, 'Epochs number must be positive'
    assert args.patience >= 0, 'Patience number must be non-negative'
    assert args.seed is None or args.seed >= 0, 'Random seed must be non-negative'
    assert 0 < args.test_part < 1, 'Test part must be in (0, 1)'
    assert 0 < args.val_part < 1, 'Test part must be in (0, 1)'
    
    if args.output is None:
        output_file = f"{args.dataset}_exp {strftime('%d_%m %H_%M_%S', gmtime())}.json"
    else:
        output_file = args.output
    
    ds_map = get_ds_map()
    ds_cls = ds_map.get(args.dataset, None)
    if ds_cls is None:
        raise AttributeError(f"Dataset {args.dataset} is not found")
    ds = ds_cls(args.val_part, args.test_part, args.seed)
    return Experiment(ds, args.iter_n, args.epochs, args.patience, 
                      args.folds_n, args.cv_iters, args.cv_jobs, 
                      args.seed, output_file)
    

if __name__ == '__main__':
    experiment = cli()
    time_stamp = time()
    experiment.run(verbose=1)
    time_elapsed = int(time() - time_stamp)
    print('Time elapsed: ', time_elapsed // 60, ' min.', time_elapsed % 60, ' sec.')
    