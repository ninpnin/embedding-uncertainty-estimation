from trainerlog import get_logger
LOGGER = get_logger("sampling")
LOGGER.info("Load modules..")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from polyagamma import random_polyagamma
import copy

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
        LOGGER.debug(f"iter {ix}")
        # Get a 5 by 1 array of PG(1, 2) variates.
        xTbeta = X @ beta
        omega = random_polyagamma(N, xTbeta)
        #print(omega)
        V_omega = get_v_omega(X, omega, sigma_prior)
        mu_omega = get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega)
        
        beta = np.random.multivariate_normal(mean=mu_omega, cov=V_omega)
        yield beta

def embedding_gibbs(e, data, rounds=10, polyagamma_iter=50, yield_every=1, freeze_params=[]):
    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    sigma_prior = np.identity(e.dimensionality) * 1.0

    for ix, turn in enumerate(turns * rounds):
        LOGGER.train(f"Flip turn: {turn}, {ix}")
        for wd in words:
            data_wd = [(j, x) for (i, j, x) in data if i == wd]
            if turn == "context":
                wd = wd + "_c"
                data_wd = [(i, x) for (i, j, x) in data if j == wd]

            X = e[[ij for (ij, x) in data_wd]].numpy()
            y = np.array([x for (ij, x) in data_wd])

            beta_init = e[wd].numpy()
            e_wd_new_samples = list(polyagamma_sampler(beta_init, X, y, sigma_prior=sigma_prior, iterations=polyagamma_iter))
            
            if wd not in freeze_params:
                e[wd] = e_wd_new_samples[-1]

        if ix % (yield_every * 2) == 0:
            e_sample = copy.deepcopy(e)
            yield e_sample

