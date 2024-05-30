import json
import pickle
import os
import sys
import argparse
import cmdstanpy
from cmdstanpy import CmdStanModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='100k_v10_d2_.json')
parser.add_argument("--model_path", type=str, default='models/sgns_normalpriors.stan')
parser.add_argument("--save_dir", type=str, default='stan_fits_map')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
save_dir = args.save_dir

# --- fit settings ---
sizes = [100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000]  # None (uses all data) or list of data sizes [100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000]
D = 2
algorithm = 'Newton' # BFGS’, ‘LBFGS’, ‘Newton’


# --- Library ---
def save_fit(fit, size, save_dir=save_dir):
    filename = os.path.join(save_dir, f'stan_fit_{size}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(fit, f)

def dir_exists_check(dir_path: str):
    if os.path.isdir(dir_path):
        response = input(f"The file '{dir_path}' already exists. Do you want to continue? (y/n): ").strip().lower()
        if response == 'y':
            return True
        else:
            return False
    return True

def create_stan_data(data, size, vocab, D):
    target_word = []
    context_word = []
    posneg_labels = []  # if the pair is pos or negative sample.

    for pair in data[:size]:
        # we add +1 because stan index starts on 1.
        target_word.append(vocab[pair['v']] + 1)
        context_word.append(vocab[pair['w']] + 1)
        posneg_labels.append(pair['x'])

    stan_data = {
        'N': len(target_word),
        'V': len(vocab),
        'D': D,
        'target_word': target_word,
        'context_word': context_word,
        'posneg_labels': posneg_labels
    }

    return stan_data

# --------

dir_flag = dir_exists_check(save_dir)
if dir_flag:
    os.makedirs(save_dir, exist_ok=True)
else:
    sys.exit()

with open(data_path) as f:
    data = json.load(f)

if sizes is None:
    sizes = [len(data['data'])]

for size in sizes:
    stan_data = create_stan_data(data['data'], size, data['vocabulary'], D=D)
    
    model = CmdStanModel(stan_file=model_path)
    
    # MAP
    # returns CmdStanMLE object...
    # Note: the optimize method finds the mode of the posterior distribution, which is MAP estimate if priors are included. MLE if no priors.
    map = model.optimize(data=stan_data, algorithm=algorithm)  # , algorithm= BFGS’, ‘LBFGS’, ‘Newton’

    save_fit(map, size) 
    print(f"Saved MAP for size {size}")
