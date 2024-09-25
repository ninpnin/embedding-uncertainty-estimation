from trainerlog import get_logger
LOGGER = get_logger("sampling")
LOGGER.info("Load modules..")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from polyagamma import random_polyagamma
import copy
import progressbar
from numba import jit
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from probabilistic_word_embeddings.models import sgns_likelihood

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

def polyagamma_sampler(beta_init, X, y, iterations=2, N=None, mu_prior=None, sigma_prior=None, return_last=False):
    """

    """
    if N is None:
        N = np.ones(len(X))
    beta = beta_init
    betas = []
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
        if not return_last:
            betas.append(beta)
    if return_last:
        return beta
    else:
        return betas

def prior_sampler(beta_init, mu_prior=None, sigma_prior=None):
    if mu_prior is None:
        mu_prior = np.zeros(beta_init.shape)
    if sigma_prior is None:
        sigma_prior = np.identity(beta_init.shape)

    return np.random.multivariate_normal(mean=mu_prior, cov=sigma_prior)

def logistic_laplace_approx(X, y, mu_prior=None, sigma_prior=None):
    """

    """
    if mu_prior is None:
        mu_prior = np.zeros(X.shape[-1])
    if sigma_prior is None:
        sigma_prior = np.identity(X.shape[-1])
    
    C = 1.0 / sigma_prior[0,0]
    clf = LogisticRegression(random_state=0, C=C).fit(X, y)
    beta = clf.coef_[0]

    omega = tf.math.sigmoid(X @ beta).numpy()
    omega = omega * (1.0 - omega)
    covariance = (omega* X.T) @ X
    return np.random.multivariate_normal(mean=beta, cov=covariance)
    #return beta


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

def embedding_gibbs(e, data, rounds=10, polyagamma_iter=50, yield_every=1, laplace_approx_limit=None, freeze_params=[]):
    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    sigma_prior = np.identity(e.dimensionality) * 1.0
    if laplace_approx_limit is None:
        laplace_approx_limit = 10000000

    data_wds_cache = {}
    X_cache = {}
    y_cache = {}
    N_wd_cache = {}

    logprobs = []

    # Preprocess data
    for wd in progressbar.progressbar(words):
        for turn in turns:
            if turn == "context":
                wd = wd + "_c"
            data_wd = get_wd_data(data, wd, turn, cache=data_wds_cache)
            N_wd_cache[wd] = len(data_wd)
            X_cache[wd] = X_cache.get(wd, tf.constant([ij for (ij, x) in data_wd]))
            y_cache[wd] = y_cache.get(wd, np.array([x for (ij, x) in data_wd]))

    for ix, turn in enumerate(turns * rounds):
        LOGGER.train(f"Flip turn: {turn}, {ix}")
        prior_count = 0
        
        for wd in progressbar.progressbar(words):
            e_theta = e.theta.numpy()
            if turn == "context":
                wd = wd + "_c"
            if N_wd_cache[wd] > 0:
                X = e[X_cache[wd]].numpy()
                beta_init = e[wd].numpy()
                y = y_cache[wd]

                if N_wd_cache[wd] < laplace_approx_limit:
                    e_wd_new_samples = polyagamma_sampler(beta_init, X, y, sigma_prior=sigma_prior, iterations=polyagamma_iter, return_last=True)
                    
                    if wd not in freeze_params:
                        e[wd] = e_wd_new_samples
                else:
                    try:
                        if wd not in freeze_params:
                            e[wd] = logistic_laplace_approx(X, y, mu_prior=None, sigma_prior=sigma_prior)
                    except Exception as error:
                        LOGGER.error(f"MAP error {error}")
                        e_wd_new_samples = polyagamma_sampler(beta_init, X, y, sigma_prior=sigma_prior, iterations=polyagamma_iter, return_last=True)
                        if wd not in freeze_params:
                            e[wd] = e_wd_new_samples
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
            data_i = tf.constant([i for (i, j, x) in data])
            data_j = tf.constant([j for (i, j, x) in data])
            data_x = tf.constant([x for (i, j, x) in data], dtype=tf.float64)

            LOGGER.info(f"Calculate log posterior for the sample...")
            ll = tf.reduce_sum(sgns_likelihood(e, data_i, data_j, x=data_x))
            posterior = ll + e.log_prob(len(data_i), len(data_i))
            logprobs.append(posterior)
            LOGGER.train(f"Log posterior for the sample: {posterior}")
            yield e_sample
        if ix % (yield_every * 100) == 0 and ix > 0:
            from matplotlib import pyplot as plt
            plt.plot(range(len(logprobs)), logprobs)
            plt.show()

