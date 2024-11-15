from trainerlog import get_logger
LOGGER = get_logger("sampling")
LOGGER.info("Load modules..")
import numpy as np
from polyagamma import random_polyagamma
import copy
import progressbar
import tensorflow as tf
LOGGER.info("Done!")
from time import perf_counter as pc

def get_v_omega_tf(X, omega, sigma_prior_inv):
    XT2 = tf.transpose(X, perm=[2,1,0])
    XTomega = (XT2 * omega)
    XTomega = tf.transpose(XTomega, perm=[2,1,0])
    V_inv = tf.linalg.matmul(X, XTomega, transpose_a=True)
    V_inv = tf.transpose(V_inv, perm=[0,2,1])
    V_inv += sigma_prior_inv
    return tf.linalg.inv(V_inv)

def get_mu_omega_tf_parenthesis(X, kappa, mu_prior, sigma_prior):
    kappa = tf.transpose(tf.expand_dims(kappa, [1]), perm=[0,2,1])
    parenthesis = tf.transpose(tf.linalg.matmul(kappa, X, transpose_a=True), perm=[0,2,1])
    mu_prior = tf.expand_dims(mu_prior, axis=1)
    mu_prior = tf.transpose(mu_prior, perm=[0,2,1])
    parenthesis2 = tf.linalg.matmul(tf.linalg.inv(sigma_prior), mu_prior, transpose_a=True)
    return parenthesis + parenthesis2

def polyagamma_sampler_tf(beta_init, X, y, iterations=2, N=None, mu_prior=None, sigma_prior=None):
    """

    """
    dtype = beta_init.dtype
    beta = beta_init
    M = X.shape[0]
    K = X.shape[-1]
    CHAINS = X.shape[0]
    
    if mu_prior is None:
        #mu_prior = np.zeros(X.shape[-1])
        mu_prior = tf.stack([tf.zeros(K, dtype=beta_init.dtype) for _ in range(M)])
    if sigma_prior is None:
        sigma_prior = tf.stack([tf.eye(K, dtype=beta_init.dtype) for _ in range(M)])
    XT = tf.transpose(X, perm=[1,0,2])
    NT = N.numpy().T
    kappa = y - N/2
    
    omega = tf.Variable(NT, dtype=dtype)
    parenthesis = get_mu_omega_tf_parenthesis(X, kappa, mu_prior, sigma_prior)
    sigma_prior_inv = tf.linalg.inv(sigma_prior)
    
    pg_tds = []
    for ix in range(iterations):
        #LOGGER.debug(f"iter {ix}")
        # Get a 5 by 1 array of PG(1, 2) variates.
        xTbeta = tf.reduce_sum(XT * beta, axis=-1)
        t0 = pc()
        omega_numpy = random_polyagamma(NT, xTbeta.numpy())
        pg_tds.append(pc() - t0)
        omega.assign(omega_numpy)
        #print(omega)
        V_omega = get_v_omega_tf(X, omega, sigma_prior_inv)
        mu_omega = tf.reduce_sum(tf.linalg.matmul(V_omega, parenthesis, transpose_a=True), axis=-1)

        # Use the square root of the matrix U S^(1/2) V^T
        # from the SVD to generate multivariate random vectors
        
        # x_prime = mu + (U S^(1/2) V^T) x

        S, U, V = tf.linalg.svd(V_omega)
        S_sqrt =  tf.linalg.diag(tf.sqrt(S))
        U_S_sqrt = tf.linalg.matmul(U, S_sqrt)
        sqrt_V_omega = tf.linalg.matmul(U_S_sqrt, V)
        epsilon = tf.random.normal([CHAINS, K, 1], dtype=dtype)
        diff = tf.linalg.matmul(sqrt_V_omega, epsilon, transpose_a=True)
        beta = mu_omega + tf.reduce_sum(diff, axis=-1)

        yield beta
    print("Polya-Gamma sampling total:", np.sum(pg_tds), "(s)")

def get_v_omega(X, omega, sigma_prior):
    # Equivalent to the following, but optimized
    # Omega = np.diag(omega)
    # V_inv = X.T @ Omega @ X
    V_inv = (omega* X.T) @ X
    V_inv += np.linalg.inv(sigma_prior)
    return np.linalg.inv(V_inv)

def get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega):
    kappa = y - N/2
    parenthesis = X.T @ kappa + np.linalg.inv(sigma_prior) @ mu_prior
    return V_omega @ parenthesis

def polyagamma_sampler(beta_init, X, y, iterations=2, N=None, mu_prior=None, sigma_prior=None):
    """

    """
    if N is None:
        N = np.ones(len(X))
    beta = beta_init
    
    if mu_prior is None:
        mu_prior = np.zeros(X.shape[-1])
    if sigma_prior is None:
        sigma_prior = np.identity(X.shape[-1])
        
    for ix in range(iterations):
        #LOGGER.debug(f"iter {ix}")
        # Get a 5 by 1 array of PG(1, 2) variates.
        xTbeta = X @ beta
        omega = random_polyagamma(N, xTbeta)
        #print(omega)
        V_omega = get_v_omega(X, omega, sigma_prior)
        mu_omega = get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega)
        
        beta = np.random.multivariate_normal(mean=mu_omega, cov=V_omega)
        yield beta

def prior_sampler(beta_init, mu_prior=None, sigma_prior=None):
    if mu_prior is None:
        mu_prior = np.zeros(beta_init.shape)
    if sigma_prior is None:
        sigma_prior = np.identity(beta_init.shape)

    return np.random.multivariate_normal(mean=mu_prior, cov=sigma_prior)

def get_wd_data(data, wd, turn, cache={}):
    if turn != "context":
        if wd in cache:
            return cache[wd]
        else:
            data_wd = [(j, x) for (i, j, x) in data if i == wd]
            cache[wd] = data_wd
            return data_wd
    else:
        if wd in cache:
            return cache[wd]
        else:
            data_wd = [(i, x) for (i, j, x) in data if j == wd]
            cache[wd] = data_wd
            return data_wd

def embedding_gibbs(e, data, rounds=10, polyagamma_iter=50, yield_every=1, freeze_params=[]):
    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    sigma_prior = np.identity(e.dimensionality) * 1.0

    data_wds_cache = {}
    for ix, turn in enumerate(turns * rounds):
        LOGGER.train(f"Flip turn: {turn}, {ix}")
        prior_count = 0
        
        for wd in progressbar.progressbar(words):
            if turn == "context":
                wd = wd + "_c"
            data_wd = get_wd_data(data, wd, turn, cache=data_wds_cache)
            #print(wd)
            if len(data_wd) > 0:
                X = e[[ij for (ij, x) in data_wd]].numpy()
                y = np.array([x for (ij, x) in data_wd])

                beta_init = e[wd].numpy()
                e_wd_new_samples = list(polyagamma_sampler(beta_init, X, y, sigma_prior=sigma_prior, iterations=polyagamma_iter))
                
                if wd not in freeze_params:
                    e[wd] = e_wd_new_samples[-1]
            else:
                prior_count += 1
                if wd not in freeze_params:
                    e[wd] = prior_sampler(e[wd].numpy(), sigma_prior=sigma_prior)
        if prior_count >= len(words) * 0.2:
            LOGGER.warning(f"sampled from prior: {prior_count} out of {len(words)}")
        else:
            LOGGER.info(f"sampled from prior: {prior_count} out of {len(words)}")
        if ix % (yield_every * 2) == 0:
            e_sample = copy.deepcopy(e)
            yield e_sample

