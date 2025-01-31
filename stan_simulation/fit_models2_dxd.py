import json
import numpy as np
import pickle
import os
import sys
from collections import defaultdict
from trainerlog import get_logger
LOGGER = get_logger("stan", splitsec=True)
from pathlib import Path

import stan  # PyStan 3
from cmdstanpy import CmdStanModel


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='../../test_100k5/1.json')
parser.add_argument("--output_dir", type=str, default='../../test_fix_res/fix/hmc/1')
parser.add_argument("--stan_model_path", type=str, default='models/sgns_normalpriors_fix_last_aggregated.stan')
parser.add_argument("--map_dir", type=str, default='../../test_fix_res/nofix/map/1' ,help="Directory containing the previously fitted unfixed MAP models.")
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--lambda0", type=float)  # If not provided, defaults to std = sqrt(1/D)
parser.add_argument("--estimator", type=str, default='hmc')
parser.add_argument("--num_chains", type=int, default=2) #2
parser.add_argument("--num_samples", type=int, default=2000) #2000
parser.add_argument("--data_lengths", type=int, nargs="+", default=[100])

args = parser.parse_args()

LOGGER.train(f"Run on args: {args}")
data_path = args.data_path
stan_model_path = args.stan_model_path
output_dir = args.output_dir
map_dir = args.map_dir

inference_type = args.estimator.lower()  # 'hmc', 'vi', 'laplace', 'map'
num_samples = args.num_samples
num_chains = args.num_chains

# Detect aggregated vs. non-aggregated from the .stan filename
use_aggregated_data = False
if stan_model_path.split('.')[-2].split('_')[-1] == 'aggregated':
    LOGGER.info('Aggregated model detected.')
    use_aggregated_data = True
else:
    LOGGER.info('No aggregation detected.')

dataset_filename = Path(data_path).stem

D = args.dim
if args.lambda0 is not None:
    lambda0 = args.lambda0
else:
    LOGGER.info('Default lambda0 = sqrt(1.0 / D)')
    lambda0 = np.sqrt(1.0 / D)

LOGGER.info(f'Using lambda0 = {lambda0}')
sizes = args.data_lengths
LOGGER.info(f"data_lengths = {sizes}")

# -- Load data once --
with open(data_path) as f:
    data = json.load(f)

data_entries = data['data']
vocabulary = data['vocabulary']
V = len(vocabulary)
LOGGER.info(f"Total length of data {len(data_entries)}")

data = None  # free space


def load_fit_by_size(size, save_dir):
    files = os.listdir(save_dir)
    for f in files:
        if f.endswith('.pkl'):
            # split on '.' => e.g. "stan_fit_map_K100_foo_500.pkl"
            # then f.split('.')[-2].split('_') => last chunk is likely '500'
            # If that equals str(size), we assume it's the correct file
            parts = f.split('.')[-2].split('_')
            if parts[-1] == str(size):
                file_path = os.path.join(save_dir, f)
                with open(file_path, 'rb') as fn:
                    fit = pickle.load(fn)
                return fit
    LOGGER.error(f"No fit of size {size} found in {save_dir}")
    return None


def create_aggregated_stan_data_dxd(data_entries, size, vocabulary, D, lambda0, fixed_context_matrix):
    aggregated_counts = defaultdict(int)
    for entry in data_entries[:size]:
        t = vocabulary[entry['v']] + 1
        c = vocabulary[entry['w']] + 1
        lbl = entry['x']
        aggregated_counts[(t, c, lbl)] += 1

    unique_target_words = []
    unique_context_words = []
    unique_posneg_labels = []
    counts = []
    for (t, c, lbl), ct in aggregated_counts.items():
        unique_target_words.append(t)
        unique_context_words.append(c)
        unique_posneg_labels.append(lbl)
        counts.append(ct)

    stan_data = {
        'lambda': lambda0,
        'U': len(unique_target_words),
        'V': len(vocabulary),
        'D': D,
        'target_word': unique_target_words,
        'context_word': unique_context_words,
        'posneg_labels': unique_posneg_labels,
        'counts': counts,
        'fixed_context_matrix': fixed_context_matrix  # <--- ADDED FOR D×D FIX
    }
    return stan_data


def save_fit(fit, size, dim, dataset_name, estimator):
    filename = os.path.join(output_dir,
        f'stan_fit_{estimator}_K{dim}_{dataset_name}_{size}.pkl')
    with open(filename, 'wb') as f_out:
        pickle.dump(fit, f_out)


def extract_word_and_context_vectors_vi(samples_pd, vocabulary, D):
    """
    VI helper: extracts the word and context vectors from the variational parameters.
    """
    #variational_samples_pd = fit.variational_sample_pd

    # Drop the first three columns: 'lp__', 'log_p__', 'log_g__'
    relevant_params = samples_pd.drop(columns=['lp__','log_p__','log_g__'] , errors='ignore')

    # Drop columns related to 'context_vectors_raw'
    raw_context_columns = [col for col in relevant_params.columns if 'context_vectors_raw' in col]
    relevant_params = relevant_params.drop(columns=raw_context_columns)

    num_samples = relevant_params.shape[0]
    V = len(vocabulary)
    
    word_vectors = np.zeros((num_samples, V, D))
    context_vectors = np.zeros((num_samples, V, D))
    
    for d in range(D):
        for n in range(V):
            word_vectors[:, n, d] = relevant_params[f'word_vectors[{n+1},{d+1}]'].values
            context_vectors[:, n, d] = relevant_params[f'context_vectors[{n+1},{d+1}]'].values
    
    word_vectors = np.transpose(word_vectors, (1, 2, 0))  # dim = (V, D, num_samples)  - reshape to match hmc output.
    context_vectors = np.transpose(context_vectors, (1, 2, 0)) 
    

    return word_vectors, context_vectors




os.makedirs(output_dir, exist_ok=True)

LOGGER.debug(f"Estimate model on subsets of sizes: {sizes}")

for size in sizes:
    #Load the corresponding MAP fit from map_dir
    LOGGER.info(f"[Size={size}] Loading MAP from {map_dir}")
    map_fit = load_fit_by_size(size, map_dir)
    if map_fit is None:
        LOGGER.error("Failed to load MAP fit. Skipping.")
        continue

    #map_params = map.stan_variables()
    _, map_context_vectors = extract_word_and_context_vectors_vi(map_fit.optimized_params_pd, vocabulary, D=D)
    #map_context_vectors = map_params['context_vectors']
    fixed_context_matrix = map_context_vectors[-D:, :]


    # Create stan data
    if use_aggregated_data:
        LOGGER.info("Create aggregated dataset for partial-fix.")
        stan_data = create_aggregated_stan_data_dxd(
            data_entries, size, vocabulary, D, lambda0, fixed_context_matrix
        )
    else:
        LOGGER.error("This script is designed for aggregated data only right now.")
        sys.exit(1)

    LOGGER.info('Data preparation complete.')
    # data_entries = None  # If you want to free memory after each loop, do it here

    #  Fits
    with open(stan_model_path, 'r') as file:
        stan_code = file.read()

    if inference_type == 'hmc':
        LOGGER.train(f"Run HMC for {num_samples} samples and {num_chains} chains")
        model = stan.build(stan_code, data=stan_data)
        fit = model.sample(num_samples=num_samples, num_chains=num_chains)
        # To reduce storage, we typically keep the main arrays only.
        # The new model has:
        #   parameters { ... context_vectors_raw ... }
        #   transformed parameters { context_vectors ... }
        # If you want 'context_vectors' directly, do fit['context_vectors'] (if it’s declared).
        # Or store the entire `fit` object for now:
        fit = {
            'word_vectors': fit['word_vectors'],
            'context_vectors': fit['context_vectors']
        }

    elif inference_type == 'vi':
        vi_iter = 5000
        algorithm = 'meanfield'
        LOGGER.train('Running VI...')
        cmd_model = CmdStanModel(stan_file=stan_model_path)
        fit = cmd_model.variational(data=stan_data, iter=vi_iter, draws=num_samples,
                                    require_converged=True, algorithm=algorithm)

    elif inference_type == 'map':
        LOGGER.train("Running MAP via cmdstanpy.optimize()")
        cmd_model = CmdStanModel(stan_file=stan_model_path)
        jacobian = True
        fit = cmd_model.optimize(data=stan_data, jacobian=jacobian)

    elif inference_type == 'laplace':
        LOGGER.train("Running Laplace (first MAP, then laplace_sample).")
        cmd_model = CmdStanModel(stan_file=stan_model_path)
        map_obj = cmd_model.optimize(data=stan_data, jacobian=True)
        fit = cmd_model.laplace_sample(data=stan_data, mode=map_obj,
                                       draws=num_samples, jacobian=True)
    else:
        LOGGER.error(f"Unknown estimator: {inference_type}")
        sys.exit(1)

    LOGGER.train("finished fit.")

    # 5) Save the fit
    save_fit(fit, size, D, dataset_filename, inference_type)
    LOGGER.train(f"Saved fit for size {size} in {output_dir}")

