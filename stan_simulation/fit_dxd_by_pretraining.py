import json
import numpy as np
import stan
import pickle
import os

"""
Fixate the top DxD context embeddings based on context embeddings from a previously fit model.

Fit unfixed model and use those as input to the new fixed.
"""

#D = 2 Determined by the loaded model fit.

data_path = '100k_v10_d2_.json'
stan_model_path = 'models/sgns_dxd.stan'
output_dir = 'stan_fit_dxdfix'
nofix_dir = 'stan_fits_test'

sizes = [100, 200] #[100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000] 

with open(data_path) as f:
    data = json.load(f)

data_entries = data['data']
vocabulary = data['vocabulary']
V = len(vocabulary)

with open(stan_model_path, 'r') as file:
    stan_code = file.read()

def create_stan_data(data_entries, size, vocabulary, fixed_context_matrix):
    target_word = []
    context_word = []
    posneg_labels = []

    for entry in data_entries[:size]:
        target_word.append(vocabulary[entry['v']] + 1)  # +1 because Stan uses 1-based indexing
        context_word.append(vocabulary[entry['w']] + 1)
        posneg_labels.append(entry['x'])

    stan_data = {
        'N': len(target_word),
        'V': len(vocabulary),
        'D': D,
        'target_word': target_word,
        'context_word': context_word,
        'posneg_labels': posneg_labels,
        'fixed_context_matrix': fixed_context_matrix
    }
    return stan_data

def save_fit(fit, size):
    filename = os.path.join(output_dir, f'stan_fit_{size}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(fit, f)



os.makedirs(output_dir, exist_ok=True)

for size in sizes:
    # -- Load the corresponding fit for the current size --
    fit_to_load = os.path.join(nofix_dir, f'stan_fit_{size}.pkl')
    with open(fit_to_load, 'rb') as f:
        fit_loaded = pickle.load(f)
    
    context_vectors = np.mean(fit_loaded['context_vectors'], axis=2)
    D = context_vectors.shape[1]
    fixed_context_matrix = context_vectors[:D, :D]
    
    # -- Fit stan model --
    stan_data = create_stan_data(data_entries, size, vocabulary, fixed_context_matrix)
    posterior = stan.build(stan_code, data=stan_data)
    fit = posterior.sample(num_samples=1000, num_chains=1)
    
    save_fit(fit, size)
    print(f"Saved fit for size {size}")

