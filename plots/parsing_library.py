import json
import itertools
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_eta(vi, wi, theta, V):
    rho_v = theta[vi, :]          # ρ_v = θ_v
    alpha_w = theta[wi + V, :]    # α_w = θ_{V + w}
    eta = rho_v.dot(alpha_w)      # η = ρ_v^T α_w
    return eta

def calculate_p(vi, wi, theta, V):
    eta = calculate_eta(vi, wi, theta, V)
    return sigmoid(eta)

def load_fit(size, save_dir):
    filename = os.path.join(save_dir, f'stan_fit_{size}.pkl')
    with open(filename, 'rb') as f:
        fit = pickle.load(f)
    return fit

def load_fit_by_size(size, save_dir):
    # this method looks for the correct file in the dir
    # less constricted than load_fit()
    files = os.listdir(save_dir)
    for f in files:
        if f[-3:] == 'pkl' and f.split('.')[-2].split('_')[-1] == str(size): #make sure if pkl and check for size.
            file_path = os.path.join(save_dir,f)
            with open(file_path, 'rb') as fn:
                fit = pickle.load( fn )
            return fit
    print(f'no fit of size {size} in {save_dir}')
    return None            


def calculate_rmse_slow(theta_samples, true_theta, vocabulary):
    V = len(vocabulary)
    
    p_true = []
    p_avg = []

    for word1 in vocabulary:
        for word2 in vocabulary:
            vi = vocabulary[word1]
            wi = vocabulary[word2]
            p_true.append(calculate_p(vi, wi, true_theta, V))

            p_samples = [calculate_p(vi, wi, theta, V) for theta in theta_samples]
            p_avg.append(np.mean(p_samples))
    
    p_true = np.array(p_true)
    p_avg = np.array(p_avg)
    
    rmse = np.sqrt(np.mean((p_true - p_avg) ** 2))
    return rmse

def calculate_rmse(theta_samples, true_theta, vocabulary):
    V = len(vocabulary)
    D = true_theta.shape[1]  # Dimensionality of the embeddings
    
    # --- true ---
    rho_v = true_theta[:V, :]    # (V, D)
    alpha_w = true_theta[V:, :]     # (V, D)
    
    E_true = np.dot(rho_v, alpha_w.T) #E_true = rho_v @ alpha_w.T  (V, V)
    
    p_true = sigmoid(E_true) 
    
    # --- sampled ---
    rho_v_samples = theta_samples[:, :V, :]  # (N, V, D)
    alpha_w_samples = theta_samples[:, V:, :]   # (N, V, D)
     
    #E_samples = rho_v_samples @ alpha_w_samples.transpose(0, 2, 1)
    E_samples = np.matmul(rho_v_samples, np.transpose(alpha_w_samples, (0, 2, 1))) # E_samples: (N, V, V)
    
    p_samples = sigmoid(E_samples)
    
    #  p_avg = mean over samples axis
    p_avg = np.mean(p_samples, axis=0)  # (V, V)
    
    rmse = np.sqrt(np.mean((p_true - p_avg) ** 2))
    return rmse


def extract_word_and_context_vectors_vi2(fit, vocabulary, D=2):
    """
    VI helper: extracts the word and context vectors from the variational parameters.
    """
    variational_samples_pd = fit.variational_params_pd

    # exclude the first three columns (lp__, log_p__, log_g__) and context_vectors_raw
    relevant_params = np.delete(variational_samples_pd, np.s_[3:3+(len(vocabulary)-D)*D], axis=1)
    
    num_samples = relevant_params.shape[0]
    word_vectors = np.zeros((num_samples, len(vocabulary), D))
    context_vectors = np.zeros((num_samples, len(vocabulary), D))
    
    for d in range(D):
        for n in range(len(vocabulary)):
            word_vectors[:, n, d] = relevant_params[:, n + d * len(vocabulary)]
            context_vectors[:, n, d] = relevant_params[:, len(vocabulary) * D + n + d * len(vocabulary)]
    
    return word_vectors, context_vectors


def extract_word_and_context_vectors_vi(samples_pd, vocabulary, D=2):
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

def extract_word_and_context_vectors_gibbs(filepath, vocabulary, D=2):
    df = pd.read_csv(filepath)
    num_samples = df.shape[0]
    V = len(vocabulary)
    
    word_vectors = np.zeros((V, D, num_samples))
    context_vectors = np.zeros((V, D, num_samples))
    
    for n in range(V):
        for d in range(D):
            word_vectors[n, d, :] = df[f'word{n}_{d}'].values
            context_vectors[n, d, :] = df[f'word{n}_c_{d}'].values
    return word_vectors, context_vectors

def credible_interval(samples, confidence=0.90):
    lower_bound = np.percentile(samples, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(samples, (1 + confidence) / 2 * 100)
    return lower_bound, upper_bound