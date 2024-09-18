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

