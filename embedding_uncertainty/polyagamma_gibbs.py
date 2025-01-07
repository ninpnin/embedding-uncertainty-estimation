from trainerlog import get_logger
LOGGER = get_logger("sampling", splitsec=True)
LOGGER.info("Load modules..")
import numpy as np
from polyagamma import random_polyagamma
import copy
import progressbar
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from probabilistic_word_embeddings.models import sgns_likelihood

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

def polyagamma_sampler_tf(beta_init, X, y, iterations=2, kappa=None, N=None, mu_prior=None, sigma_prior=None, multivariate_method="svd"):
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
    if kappa is None:
        kappa = y - N/2
    
    omega = tf.Variable(NT, dtype=dtype)
    parenthesis = get_mu_omega_tf_parenthesis(X, kappa, mu_prior, sigma_prior)
    sigma_prior_inv = tf.linalg.inv(sigma_prior)
    
    pg_tds = []
    for ix in range(iterations):
        LOGGER.debug(f"iter {ix}")
        # Get a 5 by 1 array of PG(1, 2) variates.
        xTbeta = tf.reduce_sum(XT * beta, axis=-1)
        t0 = pc()
        LOGGER.debug(f"Sample PG")
        omega_numpy = random_polyagamma(NT, xTbeta.numpy())
        pg_tds.append(pc() - t0)
        omega.assign(omega_numpy)
        #print(omega)
        LOGGER.debug(f"calculate V_omega")
        V_omega = get_v_omega_tf(X, omega, sigma_prior_inv)
        mu_omega = tf.reduce_sum(tf.linalg.matmul(V_omega, parenthesis, transpose_a=True), axis=-1)

        L = None
        if multivariate_method == "svd":
            # Use the square root of the matrix U S^(1/2) V^T
            # from the SVD to generate multivariate random vectors
            # x_prime = mu + (U S^(1/2) V^T) x

            # Batched SVD is very inefficient on TF/CUDA, use CPU instead
            LOGGER.debug(f"Do tf cpu SVD on V_omega")
            with tf.device('/cpu:0'):
                S, U, V = tf.linalg.svd(V_omega)
            LOGGER.debug(f"Calculate matrix sqrt")
            S_sqrt =  tf.linalg.diag(tf.sqrt(S))
            U_S_sqrt = tf.linalg.matmul(U, S_sqrt)
            sqrt_V_omega = tf.linalg.matmul(U_S_sqrt, V)
            L = sqrt_V_omega
        elif multivariate_method == "cholesky":
            # Use Cholesky LL^T to generate multivariate random vectors
            # x_prime = mu + L x
            LOGGER.debug(f"Do Cholesky on V_omega")
            L = tf.linalg.cholesky(V_omega)
        elif multivariate_method == "eigh":
            #eigvals, eigvecs = tf.linalg.eigh(V_omega)
            #L =
            raise NotImplementedError 

        
        LOGGER.debug(f"Sample multivariate normal")
        epsilon = tf.random.normal([CHAINS, K, 1], dtype=dtype)
        diff = tf.linalg.matmul(L, epsilon, transpose_a=True)
        beta = mu_omega + tf.reduce_sum(diff, axis=-1)

        yield beta
    LOGGER.debug(f"Polya-Gamma sampling total: {np.sum(pg_tds)} (s)")

def get_v_omega(X, omega, sigma_prior):
    # Equivalent to the following, but optimized
    # Omega = np.diag(omega)
    # V_inv = X.T @ Omega @ X
    V_inv = (omega* X.T) @ X
    V_inv += np.linalg.inv(sigma_prior)
    return np.linalg.inv(V_inv)

def get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega, kappa=None):
    if kappa is None:
        kappa = y - N/2
    parenthesis = X.T @ kappa + np.linalg.inv(sigma_prior) @ mu_prior
    return V_omega @ parenthesis

def polyagamma_sampler(beta_init, X, y, iterations=2, N=None, kappa=None, mu_prior=None, sigma_prior=None, return_last=False):
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
        mu_omega = get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega, kappa=kappa)

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

def aggregate_data(data, words, e, turns=["word", "context"]):
    X_cache, N_wd_cache, kappa_cache = {}, {}, {}

    positive_samples = {}
    total_samples = {}
    for i, j, x in data:
        positive_samples[(i,j)] = positive_samples.get((i,j), 0) + x
        total_samples[(i,j)] = total_samples.get((i,j), 0) + 1

        positive_samples[(j,i)] = positive_samples.get((j,i), 0) + x
        total_samples[(j,i)] = total_samples.get((j,i), 0) + 1

    for wd in words:
        for turn in turns:
            if turn == "context":
                wd = wd + "_c"

            X_cache_wd = []
            N_cache_wd = []
            kappa_cache_wd = []
            for v in e.vocabulary:
                if (wd, v) in total_samples:
                    # kappa = y - N/2
                    X_cache_wd.append(v)
                    N_cache_wd.append(total_samples[(wd,v)])
                    kappa_v =  positive_samples[(wd,v)] - total_samples[(wd,v)] / 2
                    kappa_cache_wd.append(kappa_v)

            X_cache[wd] = X_cache_wd
            kappa_cache[wd] = kappa_cache_wd
            N_wd_cache[wd] = N_cache_wd
    return X_cache, N_wd_cache, kappa_cache

def embedding_gibbs(e, data, rounds=10, polyagamma_iter=50, yield_every=1, lambda0=None, freeze_params=[], aggregate=True, plot=True):
    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    sigma_prior = np.identity(e.dimensionality) * 1.0
    if lambda0 is not None:
        LOGGER.info(f"Use provided lambda0: {lambda0}")
    else:
        lambda0 = e.lambda0
        LOGGER.info(f"Use lambda0 from the embedding object: {lambda0}")
    sigma_prior = sigma_prior * lambda0

    data_wds_cache = {}
    X_cache, y_cache, N_wd_cache, kappa_cache = {}, {}, {}, {}
    logprobs = []

    # Preprocess data
    if not aggregate:
        for wd in progressbar.progressbar(words):
            for turn in turns:
                if turn == "context":
                    wd = wd + "_c"
                data_wd = get_wd_data(data, wd, turn, cache=data_wds_cache)
                X_cache[wd] = X_cache.get(wd, tf.constant([ij for (ij, x) in data_wd]))
                y_cache[wd] = y_cache.get(wd, np.array([x for (ij, x) in data_wd]))
                N_wd_cache[wd] = np.ones(len(data_wd))
    else:
        X_cache, N_wd_cache, kappa_cache = aggregate_data(data, words, e)

    for ix, turn in enumerate(turns * rounds):
        LOGGER.train(f"Flip turn: {turn}, {ix}")
        prior_count = 0

        # Calculate log_posterior and yield sample
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

        # Plot log_posterior graph
        if plot and ix % (yield_every * 100) == 0 and ix > 0:
            from matplotlib import pyplot as plt
            plt.plot(range(len(logprobs)), logprobs)
            plt.show()

        # Do sampling
        for wd in progressbar.progressbar(words):
            e_theta = e.theta.numpy()
            if turn == "context":
                wd = wd + "_c"
            if sum(N_wd_cache[wd]) > 0:
                X = e[X_cache[wd]].numpy()
                beta_init = e[wd].numpy()
                kappa_wd = kappa_cache.get(wd)
                y = None
                if kappa_wd is None:
                    y = y_cache[wd]

                e_wd_new_samples = polyagamma_sampler(beta_init, X, y, sigma_prior=sigma_prior, N=N_wd_cache[wd], iterations=polyagamma_iter, return_last=True, kappa=kappa_wd)
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

def embedding_gibbs_tf(e, data, rounds=10, polyagamma_iter=50, yield_every=1, lambda0=None, freeze_params=[], aggregate=True, multivariate_method="svd"):
    if multivariate_method not in ["svd", "cholesky", "eigh"]:
        raise ValueError("'multivariate_method' should be either 'svd', 'cholesky' or 'eigh'")
    else:
        print("Use method", multivariate_method)

    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    sigma_prior = np.identity(e.dimensionality) * 1.0
    if lambda0 is not None:
        LOGGER.info(f"Use provided lambda0: {lambda0}")
    else:
        lambda0 = e.lambda0
        LOGGER.info(f"Use lambda0 from the embedding object: {lambda0}")
    sigma_prior = sigma_prior * lambda0


    data_wds_cache = {}
    #X_cache, y_cache, N_wd_cache, kappa_cache = {}, {}, {}, {}
    logprobs = []

    X_cache, N_wd_cache, kappa_cache = aggregate_data(data, words, e)

    for ix, turn in enumerate(turns * rounds):
        LOGGER.train(f"Flip turn: {turn}, {ix}")
        prior_count = 0

        # Calculate log_posterior and yield sample
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

        # Plot log_posterior graph
        if ix % (yield_every * 100) == 0 and ix > 0:
            from matplotlib import pyplot as plt
            plt.plot(range(len(logprobs)), logprobs)
            plt.show()

        # Do sampling

        block_size = 100
        blocks = len(words) // block_size
        if len(words) % block_size != 0:
            blocks += 1

        words_prime = sorted(words, key=lambda wd_i: N_wd_cache[wd_i])
        for block_ix in progressbar.progressbar(range(blocks)):

            wd = words_prime[block_ix * block_size: (1 +block_ix) * block_size]
            e_theta = e.theta.numpy()
            if turn == "context":
                wd = [f"{wd_i}_c" for wd_i in wd]

            wd_skipped = [wd_i for wd_i in wd if sum(N_wd_cache[wd_i]) < 1]
            wd = [wd_i for wd_i in wd if sum(N_wd_cache[wd_i]) >= 1]
            if len(wd) >= 1:
                X_wd = [X_cache[wd_i] for wd_i in wd]
                X_wd = tf.ragged.constant(X_wd)
                X_padded = e[X_wd].to_tensor()

                beta_init = e[wd]
                kappa_wd = tf.ragged.constant([kappa_cache.get(wd_i) for wd_i in wd])
                kappa_padded = kappa_wd.to_tensor()
                kappa_padded = tf.cast(kappa_padded, dtype=tf.float64)

                N_wd_ragged = tf.ragged.constant([N_wd_cache[wd_i] for wd_i in wd])
                N_wd_padded = N_wd_ragged.to_tensor()

                N_wd_padded = tf.math.maximum(N_wd_padded, tf.ones(N_wd_padded.shape, dtype=N_wd_padded.dtype))
                last_sample = None
                for beta_sample in polyagamma_sampler_tf(beta_init, X_padded, None, kappa=kappa_padded, sigma_prior=sigma_prior, N=N_wd_padded, iterations=polyagamma_iter, multivariate_method=multivariate_method):
                    last_sample = beta_sample
                # TODO: if wd not in freeze_params:
                e[wd] = beta_sample

            # Sample from the prior
            for wd_i in wd_skipped:
                e[wd_i] = prior_sampler(e[wd_i].numpy(), sigma_prior=sigma_prior)
            prior_count += len(wd_skipped)

        if prior_count >= len(words) * 0.2:
            LOGGER.warning(f"sampled from prior: {prior_count} out of {len(words)}")
        else:
            LOGGER.info(f"sampled from prior: {prior_count} out of {len(words)}")

