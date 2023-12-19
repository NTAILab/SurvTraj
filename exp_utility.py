import numpy as np
import json
from sys import argv

def print_summary(filename):
    with open(filename, 'r') as file:
        d = json.load(file)
        res_mtx = []
        names_list = []
        for key, val in d.items():
            names_list.append(key)
            res_mtx.append(val)
    res_mtx = np.asarray(res_mtx)
    for i, name in enumerate(names_list):
        print(name)
        print(f'\tmean: {np.mean(res_mtx[i])}, std: {np.std(res_mtx[i])}')
        mask = np.zeros(res_mtx.shape, dtype=bool)
        mask[i, :] = True
        masked_mtx = np.ma.array(res_mtx - res_mtx[None, i], mask=mask)
        print(f'\twins: {np.ma.sum(np.ma.all(masked_mtx < 0, axis=0))}/{res_mtx.shape[1]}')

DEFAULT_FILENAME = 'VetTry.json'

if __name__=='__main__':
    filename = argv[1] if len(argv) > 1 else DEFAULT_FILENAME
    print_summary(filename)
