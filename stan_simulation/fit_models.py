import json
import stan
import pickle
import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='100k_v10_d2_.json')
parser.add_argument("--model_path", type=str, default='models/sgns_normalpriors.stan')
parser.add_argument("--save_dir", type=str, default='stan_fits')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
save_dir = args.save_dir

# --- fit settings ---
sizes = None# None (uses all data) or list of data sizes [100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000]
D = 2
num_samples = 1000
num_chains = 1

# --- Library ---
def save_fit(fit, size, save_dir=save_dir):
    filename = os.path.join(save_dir, f'stan_fit_{size}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(fit, f)


def dir_exists_check(dir_path:str):
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
    posneg_labels = [] # if the pair is pos or negative sample.

    for pair in data[:size]:
        # we add +1 because stan index starts on 1.
        target_word.append(vocab[pair['v']]+1)
        context_word.append(vocab[pair['w']]+1)
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

with open(model_path, 'r') as file:
    stan_code = file.read()

if sizes is None:
    sizes = [len(data)]

for size in sizes:
    stan_data = create_stan_data(data['data'], size, data['vocabulary'], D=D)
    posterior = stan.build(stan_code, data=stan_data)
    fit = posterior.sample(num_samples=num_samples, num_chains=num_chains)
    save_fit(fit, size)
    print(f"Saved fit for size {size}")