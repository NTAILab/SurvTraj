import numpy as np
import json
from sys import argv

def print_summary(filename):
    with open(filename, 'r') as file:
        d = json.load(file)
        for key, val in d.items():
            print(key)
            print(f'\tmean: {np.mean(val)}, std: {np.std(val)}')

DEFAULT_FILENAME = 'VetTry.json'

if __name__=='__main__':
    filename = argv[1] if len(argv) > 1 else DEFAULT_FILENAME
    print_summary(filename)
