import json
import pickle
import os
import sys
from collections import defaultdict
import argparse
import cmdstanpy
from cmdstanpy import CmdStanModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='../real_data/movielens/movielens_with_ns.json') #'100k_v10_d2_.json'
parser.add_argument("--model_path", type=str, default='models/sgns_normalpriors_aggregated.stan')
parser.add_argument("--save_dir", type=str, default='movielens/movielens_map')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
save_dir = args.save_dir

# --- fit settings ---
sizes = [100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000]  # None (uses all data) or list of data sizes [100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000]
D = 2
algorithm = 'Newton' # BFGS’, ‘LBFGS’, ‘Newton’
lambda0 = 1.0

use_aggregated_data = False
if model_path.split('.')[-2].split('_')[-1] == 'aggregated':
    print('Aggregated model detected.')
    use_aggregated_data = True
else:
    print('No aggregation.')

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

def create_stan_data(data, size, vocab, D, lambda0=1.0):
    target_word = []
    context_word = []
    posneg_labels = []  # if the pair is pos or negative sample.

    for pair in data[:size]:
        # we add +1 because stan index starts on 1.
        target_word.append(vocab[pair['v']] + 1)
        context_word.append(vocab[pair['w']] + 1)
        posneg_labels.append(pair['x'])

    stan_data = {
        'lambda': lambda0,
        'N': len(target_word),
        'V': len(vocab),
        'D': D,
        'target_word': target_word,
        'context_word': context_word,
        'posneg_labels': posneg_labels
    }

    return stan_data


def create_aggregated_stan_data(data_entries, size, vocabulary, D, lambda0=1.0):
    # dict to store the counts of each unique word pair. Counts positive and negative samples seperately.
    aggregated_counts = defaultdict(int) #defualt value = 0

    for entry in data_entries[:size]:
        target_word = vocabulary[entry['v']] + 1  # +1 because Stan uses 1-based indexing
        context_word = vocabulary[entry['w']] + 1
        posneg_label = entry['x']
        
        # Count occurrences : (target_word, context_word, posneg_label)
        aggregated_counts[(target_word, context_word, posneg_label)] += 1

    unique_target_words = []
    unique_context_words = []
    unique_posneg_labels = []
    counts = []
    for (target, context, label), count in aggregated_counts.items():
        unique_target_words.append(target)
        unique_context_words.append(context)
        unique_posneg_labels.append(label)
        counts.append(count)

    stan_data = {
        'lambda': lambda0,
        'U': len(unique_target_words),  # number of unique pairs
        'V': len(vocabulary),
        'D': D,
        'target_word': unique_target_words,
        'context_word': unique_context_words,
        'posneg_labels': unique_posneg_labels,
        'counts': counts,
        #'fixed_context_matrix': fixed_context_matrix
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

data_entries = data['data']
vocabulary = data['vocabulary']
V = len(vocabulary)

for size in sizes:
    print(size)
    if use_aggregated_data: 
        stan_data = create_aggregated_stan_data(data_entries, size, vocabulary, D=D, lambda0=lambda0)
    else:
        stan_data = create_stan_data(data_entries, size, vocabulary, D=D, lambda0=lambda0)
    print('done data')
    print(stan_data['U'], stan_data['V'])
    model = CmdStanModel(stan_file=model_path)
    print('model setup')   
    # MAP
    # returns CmdStanMLE object...
    # Note: the optimize method finds the mode of the posterior distribution, which is MAP estimate if priors are included. MLE if no priors.
    map = model.optimize(data=stan_data, algorithm=algorithm)  # , algorithm= BFGS’, ‘LBFGS’, ‘Newton’

    save_fit(map, size) 
    print(f"Saved MAP for size {size}")
