import json
import numpy as np
import pandas as pd
import pickle
import os
import sys
from collections import defaultdict
from trainerlog import get_logger
LOGGER = get_logger("stan", splitsec=True)
from pathlib import Path

import stan
from cmdstanpy import CmdStanModel

"""
Fixate the top DxD context embeddings based on context embeddings from a previously fit model.

Fit unfixed MAP and use those as input to the new fixed.
NOTE: MAP estimate is only supported by CmdStanPy.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='../../us-congress/congress-stemmed-ws-2-ns-1-vocab-2000-0.json') #'100k_v10_d2_.json' ten_2d_datasets/1.json
parser.add_argument("--output_dir", type=str, default='test') #stan_fits_dxd_mapjacobian
parser.add_argument("--stan_model_path", type=str, default='models/sgns_normalpriors_aggregated.stan')
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--lambda0", type=float) # default=1.0
parser.add_argument("--estimator", type=str, default='hmc')
parser.add_argument("--num_chains", type=int, default=2)
parser.add_argument("--num_samples", type=int, default=2000)
parser.add_argument("--data_lengths", type=int, nargs="+", default=[100, 200])
args = parser.parse_args()

SAVE_FULL_HMC = False # True takes a lot of disk space. only applicable to hmc.

LOGGER.train(f"Run on args: {args}")
data_path = args.data_path
stan_model_path = args.stan_model_path
output_dir = args.output_dir

inference_type = args.estimator.lower() # 'vi' # hmc, vi, laplace, map
num_samples = args.num_samples #I use 1k in the current plots/results.
num_chains = args.num_chains # for HMC

use_aggregated_data = True # should never not use aggregated really.
if stan_model_path.split('.')[-2].split('_')[-1] == 'aggregated':
    LOGGER.info('Aggregated model detected.')
    use_aggregated_data = True
else:
    LOGGER.info('No aggregation.')


dataset_filename = Path(data_path).stem

D = args.dim # embedding dimension.
if args.lambda0 is not None:
    lambda0 = args.lambda0
else:
    LOGGER.info('Default lambda0')
    lambda0 = np.sqrt(1.0 / D)  # Default value

LOGGER.info(f'Using lambda0 = {lambda0}')



#[100, 200, 500, 
sizes = args.data_lengths # todo extract from nofix_dir
LOGGER.info(f"data_lengths = {sizes}")

with open(data_path) as f:
    data = json.load(f)

data_entries = data['data']
vocabulary = data['vocabulary']
V = len(vocabulary)

if sizes is None:
    sizes = [len(data_entries)]

LOGGER.info(f"Total length of data {len(data_entries)}")

data = None # remove from memory

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

def save_fit(fit, size, dim, dataset_name, estimator):
    try:
        reduced_size = sys.getsizeof(fit)
        LOGGER.info(f"fit object size (approx): {reduced_size / (1024 ** 2):.2f} MB")
    except:
        print('Couldnt log fit size')

    filename = os.path.join(output_dir, f'stan_fit_{estimator}_K{dim}_{dataset_name}_{size}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(fit, f)

def load_fit(size, save_dir):
    # TODO: match save_fit
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

#dir_flag = dir_exists_check(output_dir)
#if dir_flag:
#    os.makedirs(output_dir, exist_ok=True)
#else:
#    sys.exit()




os.makedirs(output_dir, exist_ok=True)

LOGGER.debug(f"Estimate model on subsets of sizes: {sizes}")
for size in sizes:
    # -- Load the corresponding MAP fit for the current size --
    #map_fit_to_load = os.path.join(nofix_dir, f'stan_fit_{size}.pkl')
    #with open(map_fit_to_load, 'rb') as f:
    #    map = pickle.load(f)
    
    #map_params = map.stan_variables()
    #map_context_vectors = map_params['context_vectors']
    #D = map_context_vectors.shape[1]
    #fixed_context_matrix = map_context_vectors[:D,:] #:D

    
    # -- Fit stan model --


    if use_aggregated_data:
        LOGGER.info(f"Create aggregated dataset...")
        stan_data = create_aggregated_stan_data(data_entries, size, vocabulary, D=D, lambda0=lambda0)
    else:
        LOGGER.info(f"Create non-aggregated dataset...")
        stan_data = create_stan_data(data_entries, size, vocabulary, D=D, lambda0=lambda0)
    LOGGER.info('Data preparation complete.')

    data_entries = None #remove from memory

    if inference_type=='hmc':
        LOGGER.train(f"Run HMC for {num_samples} samples and {num_chains} chains")
        model = stan.build(stan_code, data=stan_data)
        fit = model.sample(num_samples=num_samples, num_chains=num_chains, save_warmup=False)
        if not SAVE_FULL_HMC:
            LOGGER.info('Reduced.')
            fit = {'word_vectors':fit['word_vectors'], 'context_vectors':fit['context_vectors']} ## To reduce disk storage only store nessecary.

        #LOGGER.debug(f"Model summary:\n{model.summary()}")
        # LOGGER.debug(f"{fit.to_frame().columns}")
    elif inference_type=='cmd_hmc':
        LOGGER.train(f"Run CMD HMC for {num_samples} samples and {num_chains} chains")
        model = CmdStanModel(stan_file=stan_model_path)
        
        fit = model.sample(
            data=stan_data,
            chains=num_chains,
            parallel_chains=num_chains,
            iter_sampling=num_samples,
            iter_warmup=1000,
            #save_warmup=False,
            output_dir=output_dir
        )
        
        LOGGER.info("Fit ok.")
        

    elif inference_type=='vi':
        vi_iter = 5000 #?
        algorithm = 'meanfield'
        LOGGER.train('Running VI...')
        model = CmdStanModel(stan_file=stan_model_path)
        fit = model.variational(data=stan_data, iter=vi_iter, draws=num_samples, require_converged=True, algorithm=algorithm) #algorithm='meanfield', 'fullrank'
    
    elif inference_type=='map':
        model = CmdStanModel(stan_file=stan_model_path)
        #algorithm = 'Newton'
        jacobian = True #needed for laplace.
        fit = model.optimize(data=stan_data, jacobian=jacobian) #, algorithm=algorithm

    elif inference_type=='laplace':
        #print(size)
        model = CmdStanModel(stan_file=stan_model_path) # the saved map needs to come from the same model.

        #saved_map_dir = 'stan_fits_dxd_mapjacobian'
        #print(f'loading map from {saved_map_dir}')
        #map = load_fit(size, saved_map_dir)

        map = model.optimize(data=stan_data, jacobian=True) 
        #save_fit(map, f'map_{size}')

        fit = model.laplace_sample(data=stan_data, mode=map, draws=num_samples, jacobian=True)
    
    LOGGER.train("finished fit.")
    save_fit(fit, size, D, dataset_filename, inference_type)
    LOGGER.train(f"Saved fit for size {size} in {output_dir}")
    

"""
map_filename = os.path.join(output_dir, f'stan_fit_{size}.pkl')

    with open(map_filename, 'rb') as f:
        map = pickle.load(f)

    map_params = map.stan_variables()
    map_theta = np.vstack((map_params['word_vectors'], map_params['context_vectors']))
    map_corr = calculate_true_correlation(word1, word2, vocabulary, map_theta)  #todo "true correlation" is not quite right here. But its the same function ofc.
    # -----

"""
