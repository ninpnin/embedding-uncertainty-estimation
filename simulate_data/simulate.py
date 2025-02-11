import os
import sys
import json
import argparse
import numpy as np
#import tensorflow as tf

def sigmoid(x:float): # might be a good idea to have this in a common library.
  return 1 / (1 + np.exp(-x))

def simulate_data(vocab_size:int=10, dimensionality:int=3, n_datapoints:int=100, eps_sd:float=1.0, seed:int=None, save_path:str=None, theta_seed:int=None): # sd:float=0.2
    """
    Main function for simulating artificial data.

    - theta_seed: if theta seed ignore seed and only apply to theta
    - seed: im not 100% sure this works all the way.
    """
    vocabulary = create_vocabulary(vocab_size=vocab_size)
    if theta_seed:
        print('Using theta_seed:', theta_seed)
        theta = sample_theta(vocabulary=vocabulary, dimensionality=dimensionality, eps_sd=eps_sd, seed=theta_seed)
        data = simulate_from_theta(vocabulary, theta, n_datapoints=n_datapoints,  seed=None)
    else:
        print('Using seed:', seed)
        theta = sample_theta(vocabulary=vocabulary, dimensionality=dimensionality, eps_sd=eps_sd, seed=seed)
        data = simulate_from_theta(vocabulary, theta, n_datapoints=n_datapoints,  seed=seed)

    if save_path:
        json_ = {"data": data, "theta": theta.tolist(), "vocabulary": vocabulary, "seed": (seed if seed is not None else "None"), "theta_seed": (theta_seed if theta_seed is not None else "None"), 'eps_sd': eps_sd}
        with open(save_path, 'w') as file:
            json.dump(json_, file, indent=4)
            print(f'Saved to: {save_path}')

    return data, theta, vocabulary


def load_simulated_data_generator(file_path, batch_size: int=None, max_num_observations:int=None):
    """
    If batch_size is None it yields all data in one batch
    """
    
    data = load(file_path) 
    while True:
        for item in simulated_data_generator(data, batch_size, max_num_observations):
            yield item

def load(file_path:str):
    # Side note: can use ijson for more effiecient, incremental data loading.
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def simulated_data_generator(data: dict, batch_size:int=None, max_num_observations:int=None):
    """
    Input is data or saved json from simulate_data().

    If batch_size is None it yields all data in one batch

    max_num_observations, cuts off the data and only uses the max_num_observations first data points.
    """
    if 'data' in data:
        data = data['data']

    if max_num_observations and max_num_observations < len(data):
        data = data[0:max_num_observations]

    batch_v = []
    batch_w = []
    batch_x = []
    while True:
        for item in data:
            v, w, x = item.values()
            w += "_c"
            batch_v.append(v)
            batch_w.append(w)
            batch_x.append(x)

            if batch_size is not None and len(batch_v) == batch_size:
                yield (
                    tf.constant(batch_v, dtype=tf.string),
                    tf.constant(batch_w, dtype=tf.string),
                    tf.constant(batch_x, dtype=tf.float64)
                )
                batch_v, batch_w, batch_x = [], [], []

        # remaining items as a batch
        if batch_size is None or batch_v:
            yield (
                tf.constant(batch_v, dtype=tf.string),
                tf.constant(batch_w, dtype=tf.string),
                tf.constant(batch_x, dtype=tf.float64)
            )


def create_vocabulary(vocab_size:int=10):
    words = {}
    for i in range(vocab_size):
        words['word%i'%i] = i
    return words

def sample_theta(vocabulary, dimensionality, eps_sd, seed:int=None):
    sd = eps_sd/np.sqrt(float(dimensionality))

    return np.random.default_rng(seed=seed).normal(loc=0.0, scale=sd, size=(2*len(vocabulary), dimensionality)) # scale - standard deviation.

def simulate_from_theta(vocabulary, theta, n_datapoints, seed:int=None):
    
    data = [] # format : [{"v":"some_word", "w":"another_word", "x":1}, ...]
    words = list(vocabulary.keys())
    V = len(vocabulary)
    rng = np.random.default_rng(seed=seed)
    for _ in range(n_datapoints):
        data_i = {}
        v, w = rng.choice(words, size=2, replace=False)  # Select two words v and w randomly
        data_i["v"], data_i["w"] = v, w

        rho_v = theta[vocabulary[v], :]       # $\rho_v = \theta_v$
        alpha_w = theta[vocabulary[w]+V, :]   # $\alpha_w = \theta_{V + w}$
        eta =  rho_v.dot(alpha_w)             # $\eta = \rho_v^T \alpha_w$
        p = sigmoid(eta)                      # Run it through the link function $\sigma; p = \sigma(\eta)$
        x = rng.binomial(n=1, p=p, size=1)[0] # sample from Bernoulli(p)
        data_i["x"] = int(x)
        
        data.append(data_i)

    return data

def file_exists_check(file_path:str):
    if os.path.isfile(file_path):
        response = input(f"The file '{file_path}' already exists. Do you want to continue? (y/n): ").strip().lower()
        if response == 'y':
            return True
        else:
            print("Interrupted.")
            return False
    return True

if __name__ == '__main__':
    # Observe that using the same seed (and vocab, dim etc) but changing n_datapoints
    # from ex 5 to 7, will produce the same datapoints in the 5 first in both cases.
    
    # Example usage
    # python simulate.py --save_path "test.json" --vocab_size 10 --dimensionality 3 --n_datapoints 500 --seed 1
    # python simulate.py --vocab_size 10 --dimensionality 3 --n_datapoints 20 --seed 1
    # python simulate.py

    parser = argparse.ArgumentParser(description='Simulate Data')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the simulated data. If unspecified will print the result.')
    parser.add_argument('--vocab_size', type=int, default=10, help='Vocabulary size (V)')
    parser.add_argument('--dimensionality', type=int, default=3, help='Dimension of embedding (K)')
    parser.add_argument('--n_datapoints', type=int, default=10, help='n_datapoints in the artifical dataset')
    parser.add_argument('--eps_sd', type=float, default=1.0, help='standard deviation factor of embedding elements. std = eps_std/dimensionality')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--theta_seed', type=int, default=None, help='seed for theta.')

    args = parser.parse_args()
    save_path = args.save_path
    if save_path and not file_exists_check(save_path):
        sys.exit()

    args = {k:v for k,v in vars(args).items() if (v is not None)}
    print('args:', args)
    data, theta, vocabulary = simulate_data(**args)

    if not save_path:
        print(data)
        print()
        print(theta)
        print()
        print(vocabulary)

    
  