import numpy as np
from abc import ABC, abstractstaticmethod
from surv_traj.utility import TYPE, sksurv_loader
from typing import Optional, Tuple, Dict
from sklearn.model_selection import train_test_split

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
