import json
import numpy as np
import pickle
import os
import sys
from collections import defaultdict

import stan
from cmdstanpy import CmdStanModel

"""
Fixate the top DxD context embeddings based on context embeddings from a previously fit model.

Fit unfixed MAP and use those as input to the new fixed.
NOTE: MAP estimate is only supported by CmdStanPy.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='ten_2d_datasets/1.json') #'100k_v10_d2_.json'
parser.add_argument("--output_dir", type=str, default='ten_2d_datasets/hmc/test') #stan_fits_dxd_mapjacobian
parser.add_argument("--nofix_dir", type=str, default='ten_2d_datasets/map/1')
parser.add_argument("--stan_model_path", type=str, default='models/sgns_dxd_aggregated.stan')
parser.add_argument("--lambda0", type=float, default=1.0)
args = parser.parse_args()

data_path = args.data_path
stan_model_path = args.stan_model_path
output_dir = args.output_dir
nofix_dir = args.nofix_dir

inference_type = 'hmc' # hmc, vi, laplace, map
inference_type = inference_type.lower()
lambda0 = 1.0
num_samples = 2000#1000 I use 1k in the current plots/results.

use_aggregated_data = True
if stan_model_path.split('.')[-2].split('_')[-1] == 'aggregated':
    print('Aggregated model detected.')
    use_aggregated_data = True
else:
    print('No aggregation.')


#D = 2 Determined by the loaded model fit.
sizes = [10000]#[100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000] # todo extract from nofix_dir

with open(data_path) as f:
    data = json.load(f)

data_entries = data['data']
vocabulary = data['vocabulary']
V = len(vocabulary)

with open(stan_model_path, 'r') as file:
    stan_code = file.read()

def create_stan_data(data_entries, size, vocabulary, fixed_context_matrix, D, lambda0=1.0):
    target_word = []
    context_word = []
    posneg_labels = []

    for entry in data_entries[:size]:
        target_word.append(vocabulary[entry['v']] + 1)  # +1 because Stan uses 1-based indexing
        context_word.append(vocabulary[entry['w']] + 1)
        posneg_labels.append(entry['x'])

    stan_data = {
        'lambda': lambda0,
        'N': len(target_word),
        'V': len(vocabulary),
        'D': D,
        'target_word': target_word,
        'context_word': context_word,
        'posneg_labels': posneg_labels,
        'fixed_context_matrix': fixed_context_matrix
    }
    return stan_data



def create_aggregated_stan_data(data_entries, size, vocabulary, fixed_context_matrix, D, lambda0=1.0):
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
        'fixed_context_matrix': fixed_context_matrix
    }

    return stan_data

def save_fit(fit, size):
    filename = os.path.join(output_dir, f'stan_fit_{size}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(fit, f)

def load_fit(size, save_dir):
    filename = os.path.join(save_dir, f'stan_fit_{size}.pkl')
    with open(filename, 'rb') as f:
        fit = pickle.load(f)
    return fit

def dir_exists_check(dir_path:str):
    if os.path.isdir(dir_path):
        response = input(f"The folder '{dir_path}' already exists. Do you want to continue? (y/n): ").strip().lower()
        if response == 'y':
            return True
        else:
            return False
    return True

dir_flag = dir_exists_check(output_dir)
if dir_flag:
    os.makedirs(output_dir, exist_ok=True)
else:
    sys.exit()


if sizes is None:
    sizes = [len(data['data'])]

os.makedirs(output_dir, exist_ok=True)

for size in sizes:
    # -- Load the corresponding MAP fit for the current size --
    map_fit_to_load = os.path.join(nofix_dir, f'stan_fit_{size}.pkl')
    with open(map_fit_to_load, 'rb') as f:
        map = pickle.load(f)
    
    map_params = map.stan_variables()
    map_context_vectors = map_params['context_vectors']
    D = map_context_vectors.shape[1]
    fixed_context_matrix = map_context_vectors[:D,:] #:D

    
    # -- Fit stan model --


    if use_aggregated_data:
        stan_data = create_aggregated_stan_data(data_entries, size, vocabulary, fixed_context_matrix, D=D, lambda0=lambda0)
    else:
        stan_data = create_stan_data(data_entries, size, vocabulary, fixed_context_matrix, D=D, lambda0=lambda0)
    

    if inference_type=='hmc':
    
        model = stan.build(stan_code, data=stan_data) 
        fit = model.sample(num_samples=num_samples, num_chains=1)
        #print(model.summary())
        print(fit.to_frame().columns)
    elif inference_type=='vi':
        vi_iter = 5000
        algorithm = 'meanfield'

        model = CmdStanModel(stan_file=stan_model_path)
        fit = model.variational(data=stan_data, iter=vi_iter, draws=num_samples, require_converged=True, algorithm=algorithm) #algorithm='meanfield', 'fullrank'
    
    elif inference_type=='map':
        model = CmdStanModel(stan_file=stan_model_path)
        #algorithm = 'Newton'
        jacobian = True #needed for laplace.
        fit = model.optimize(data=stan_data, jacobian=jacobian) #, algorithm=algorithm

    elif inference_type=='laplace':
        print(size)
        model = CmdStanModel(stan_file=stan_model_path) # the saved map needs to come from the same model.

        #saved_map_dir = 'stan_fits_dxd_mapjacobian'
        #print(f'loading map from {saved_map_dir}')
        #map = load_fit(size, saved_map_dir)

        map = model.optimize(data=stan_data, jacobian=True) 
        save_fit(map, f'map_{size}')

        fit = model.laplace_sample(data=stan_data, mode=map, draws=num_samples, jacobian=True)
    

    save_fit(fit, size)
    print(f"Saved fit for size {size}")
print(args)
    

"""
map_filename = os.path.join(output_dir, f'stan_fit_{size}.pkl')

    with open(map_filename, 'rb') as f:
        map = pickle.load(f)

    map_params = map.stan_variables()
    map_theta = np.vstack((map_params['word_vectors'], map_params['context_vectors']))
    map_corr = calculate_true_correlation(word1, word2, vocabulary, map_theta)  #todo "true correlation" is not quite right here. But its the same function ofc.
    # -----

"""
