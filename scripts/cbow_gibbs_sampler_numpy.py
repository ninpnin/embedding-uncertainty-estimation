import numpy as np
from polyagamma import random_polyagamma
#import os
import json
#import tensorflow as tf
import tqdm
from probabilistic_word_embeddings.embeddings import Embedding
import bidict

def cbow_gibbs_sampler(w_idx, C_idx, x_vec,
                    *, n_samples, S, V, K):
    """
    S : number of inner samples
    """

    rho_samples = np.zeros((n_samples, V, K))
    alpha_samples = np.zeros((n_samples, V, K))

    rhos = np.random.randn(V, K) / np.sqrt(K)
    alphas = np.random.randn(V, K) / np.sqrt(K)
    for it in tqdm.tqdm(range(n_samples)):
        # --- rho updates ---
        # For each unique word rho_i we can consider all the occurances at the same time.
        for wi in np.unique(w_idx):
            
            idx_i = np.where(w_idx==wi)[0]
            if len(idx_i)==0:
                # Sample from prior if no data
                rhos[wi] = np.random.randn(K) / np.sqrt(lam)
            else:
                # Otherwise, sample from posterior
                #C_idx_i = C_idx[idx_i,:]
                A = np.stack([alphas[C_idx[i]].sum(axis=0) for i in idx_i])
                kappa = x_vec[idx_i] - 0.5

                for _ in range(S):
                    eta = A @ rhos[wi]
                    omega = random_polyagamma(1, eta)
                    AOA = A.T @ (A * omega[:, None])#A.T @ np.diag(omega) @ A
                    Vw = np.linalg.inv(AOA + lam*np.eye(K))
                    mw = Vw @ (A.T @ kappa)

                    L = np.linalg.cholesky(Vw)
                    rhos[wi] = mw + L @ np.random.randn(K)

        # --- alpha updates ---
        # one at a time
        for vi in range(V): 

            idx = [i for i, Ci in enumerate(C_idx) if vi in Ci]
            if len(idx)==0:
                alphas[vi] = np.random.randn(K) / np.sqrt(lam)
                #continue
            else:
                # we look through entire vocab but only look for context words
                R = rhos[ [w_idx[i] for i in idx] ]
                # - delta -
                alpha_C = np.stack([alphas[C_idx[i]].sum(axis=0) for i in idx])
                alpha_C_v = alpha_C - alphas[vi]
                delta = np.einsum('ij,ij->i', R, alpha_C_v) #scalar prod per instance
                #   ---
                kappa = x_vec[idx] - 0.5
                
                for _ in range(S):
                    eta = R @ alphas[vi] + delta
                    omega = random_polyagamma(1, eta)
                    ROR = R.T @ (R * omega[:, None])
                    
                    Vv = np.linalg.inv(lam * np.eye(K) + ROR)
                    mv = (R.T @ (kappa - omega*delta)) @ Vv # b=0

                    L = np.linalg.cholesky(Vv)
                    alphas[vi] = mv + L @ np.random.randn(K)

        rho_samples[it] = rhos
        alpha_samples[it] = alphas
        
    return rho_samples, alpha_samples

if __name__ == '__main__':
    N, V, K = 1000, 100, 10
    datafile = f'tests/data/data-cbow-K-{K}-V-{V}-N-100000.json'
    #TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), datafile)
    with open(datafile) as f:
        docs = json.load(f)

    # --- Data setup ---
    N = 1000

    vocab = set()
    # Aggregate data into matrices
    ww = []
    CC = []
    xx = []
    for elem in docs[:N]:
        w_i, C_i, x_i = elem["w"], elem["C"], elem["x"]
        ww.append(w_i)
        CC.append(C_i)
        xx.append(x_i)

    vocab = set(ww)
    V = len(vocab)

    word2id = bidict.bidict({wd: ix for ix, wd in enumerate(vocab)})

    w_idx = np.array([word2id[w] for w in ww], dtype=int)
    C_idx = np.array([[word2id[v] for v in C_i] for C_i in CC])
    x_vec = np.array(xx, dtype=float)

    # --- SAMPLER SETUP ----
    K = 10
    lam = 1/np.sqrt(K)
    
    n_samples = 10
    S = 2

    rho_samples, alpha_samples = cbow_gibbs_sampler(w_idx, C_idx, x_vec,
                                    n_samples=n_samples, S=S, V=V, K=K)

    words = [word2id.inv[ix] for ix in range(V)]
    contexts = [wd + "_c" for wd in words]
    for rho, alpha in zip(rho_samples, alpha_samples):
        e_sample = Embedding(set(vocab), dimensionality=K)
        e_sample[words] = rho
        e_sample[contexts] = alpha

        #print(e_sample[word2id.inv[0]], rho[0])
        #print(e_sample[word2id.inv[10]], rho[10])
