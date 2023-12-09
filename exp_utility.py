import numpy as np
import json

def print_summary(filename):
    with open(filename, 'r') as file:
        d = json.load(file)
        for key, val in d.items():
            print(key)
            print(f'\tmean: {np.mean(val)}, std: {np.std(val)}')
            
if __name__=='__main__':
    filename = 'Veterans_1.json'
    print_summary(filename)
