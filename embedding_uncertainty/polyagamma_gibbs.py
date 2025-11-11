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
import networkx as nx
import tqdm

LOGGER.info("Done!")
from time import perf_counter as pc

def print_tf_tensor(arr, precision=2):
    print(np.array2string(arr.numpy(), precision=precision, suppress_small=True))

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

def polyagamma_sampler_tf(beta_init, X, y, iterations=2, kappa=None, N=None, mu_prior=None, sigma_prior=None, multivariate_method="cholesky"):
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
    assert len(data[0]) <= 4 and len(data[0]) >= 3, "The data should consist of tuples of length 3 or 4"
    counts_included = len(data[0]) == 4

    X_cache, N_wd_cache, kappa_cache = {}, {}, {}

    positive_samples = {}
    total_samples = {}
    for data_tuple in tqdm.tqdm(data):
        i, j, x = data_tuple[:3]
        count = 1
        if counts_included:
            data_tuple[-1]

        positive_samples[(i,j)] = positive_samples.get((i,j), 0) + x * count
        total_samples[(i,j)] = total_samples.get((i,j), 0) + count

        positive_samples[(j,i)] = positive_samples.get((j,i), 0) + x * count
        total_samples[(j,i)] = total_samples.get((j,i), 0) + count

    for wd in tqdm.tqdm(words):
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

    del positive_samples
    del total_samples
    
    return X_cache, N_wd_cache, kappa_cache

def embedding_gibbs(e, data, rounds=10, polyagamma_iter=50, yield_every=1, lambda0=None, freeze_params=[], aggregate=True, plot=True, ll_every=1):
    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    sigma_prior = np.identity(e.dimensionality) * 1.0
    if lambda0 is not None:
        LOGGER.info(f"Use provided lambda0: {lambda0}")
    else:
        lambda0 = e.lambda0
        LOGGER.info(f"Use lambda0 from the embedding object: {lambda0}")

    # lambda0 is the precision;
    # here we need the covariance so we divide
    sigma_prior = sigma_prior / lambda0

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
        if ix % (yield_every * ll_every * 2) == 0:
            e_sample = copy.deepcopy(e)
            data_i = tf.constant([i for (i, j, x) in data])
            data_j = tf.constant([j for (i, j, x) in data])
            data_x = tf.constant([x for (i, j, x) in data], dtype=tf.float64)

            LOGGER.info(f"Calculate log posterior for the sample...")
            ll = tf.reduce_sum(sgns_likelihood(e, data_i, data_j, x=data_x))
            posterior = ll + e.log_prob(len(data_i), len(data_i))
            logprobs.append(posterior)
            LOGGER.train(f"Log posterior for the sample: {posterior}")
            LOGGER.train(f"Avg log likelihood for the sample: {ll / len(data_x)}")
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

def split_into_independent_sets(e=None, g=None, wordcounts=None, max_size=100, verify=False):
    """
    Splits the nodes in a graph into independent sets in a greedy fashion.
    This is needed for the Gibbs sampler so that each Gibbs sample has conditionally
    independent priors.

    max_size determines the maximum size of each independent set

    """
    if g is None and not hasattr(e, 'graph'):
        words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
        no_of_blocks = len(words) // max_size
        if len(words) % max_size != 0:
            no_of_blocks += 1
        if wordcounts is not None:
            words = sorted(words, key=lambda wd_i: wordcounts[wd_i])
        blocks = []
        for block_ix in range(no_of_blocks):
            wd = words[block_ix * max_size: (1 + block_ix) * max_size]
            blocks.append(wd)

        return blocks
    elif g is None:
        g = e.graph
        for wd in e.vocabulary:
            if "_c" not in wd:
                g.add_node(wd)

    connected_components = list(nx.connected_components(g))
    ind_sets_comps = []
    disconnected = []
    #print(connected_components)
    for comp in connected_components:
        if len(comp) >= 2:
            ind_sets_comp = []
            while len(comp) >= 1:
                subg = nx.subgraph(g, comp)
                maxind = set(nx.maximal_independent_set(subg))
                #print(maxind)

                comp = [elem for elem in comp if elem not in maxind]
                ind_sets_comp.append(maxind)
                #exit()
            ind_sets_comp = sorted(ind_sets_comp, key=lambda x: -len(x))
            #print(ind_sets_comp)
            ind_sets_comps.append(ind_sets_comp)
        else:
            disconnected.append(list(comp)[0])

    ind_sets = []
    while len(ind_sets_comps) >= 1:
        new_indsets = [setlist[0] for setlist in ind_sets_comps]
        ind_sets.append(set().union(*new_indsets))
        ind_sets_comps = [setlist[1:] for setlist in ind_sets_comps if len(setlist) >= 2]

    capped_sets = []
    for elem in disconnected:
        ind_sets[0].add(elem)

    for s in ind_sets:
        if len(s) <= max_size:
            capped_sets.append(s)
        else:
            no = len(s) // max_size if len(s) % max_size == 0 else len(s) // max_size + 1
            s_list = list(s)
            if wordcounts is not None:
                s_list = sorted(s_list, key=lambda wd_i: wordcounts[wd_i])
            for ix in range(no):
                s_ix = set(s_list[ix * max_size: (ix+1) * max_size])
                capped_sets.append(s_ix)
    
    if verify:
        assert sum([len(s) for s in capped_sets]) <= len(g.nodes), "No node should be included in multiple indepdendent sets"
        assert len(set().union(*capped_sets)) >= len(g.nodes), "All nodes should be included"
    return capped_sets

def _replace_nan(t):
    indices = tf.where(tf.math.is_nan(t))
    newzeros = tf.zeros((tf.shape(indices)[0]), dtype=t.dtype)
    return tf.tensor_scatter_nd_update(t, indices, newzeros)

def get_means(e, ragged_edges):
    raw_means = tf.reduce_mean(e[ragged_edges])
    return _replace_nan(raw_means)

# The conditional Laplacian prior is a Gaussian with
# mu = len(rho_E) lambda1 / (lambda0 + lambda1 * len(rho_E) ) mean(rho_E)
# where mean(rho_E) is the mean vector of the connected edges
# and a diagonal Sigma with the variance 1/( lambda0 + lambda1 * len(rho_E))
def get_laplacian_mu(e, edges=None, edgecounts=None):
    if edges is None:
        # Without the Laplacian, default to zero mean
        return None
    else:
        assert edgecounts is not None
        scaling = (edgecounts * e.lambda1) / (e.lambda0 + e.lambda1 * edgecounts)
        mu_prior_wd = tf.transpose(tf.reduce_mean(e[edges], axis=1))
        nan_indices = tf.where(tf.math.is_nan(mu_prior_wd))
        mu_prior_wd = tf.tensor_scatter_nd_update(
            mu_prior_wd,
            nan_indices,
            tf.zeros((tf.shape(nan_indices)[0]), dtype=mu_prior_wd.dtype)
        )
        mu_prior_wd = mu_prior_wd * scaling
        return tf.transpose(mu_prior_wd)

def get_laplacian_sigma(e, M, dtype, edgecounts=None):
    K = e.dimensionality
    if edgecounts is None:
        # Without the Laplacian, the variance is 1.0 / lambda0
        return tf.stack([tf.eye(K, dtype=dtype) for _ in range(M)]) / e.lambda0
    else:
        lambdas = [e.lambda0 + e.lambda1 * ec for ec in edgecounts]
        return tf.stack([tf.eye(K, dtype=dtype) / lambdas[ix] for ix in range(M)])


def embedding_gibbs_tf(e, data, rounds=10, polyagamma_iter=50, yield_every=1, lambda0=None,
                        freeze_params=[], aggregate=True, multivariate_method="cholesky", plot=True,
                        ll_every=1):
    if multivariate_method not in ["svd", "cholesky", "eigh"]:
        raise ValueError("'multivariate_method' should be either 'svd', 'cholesky' or 'eigh'")
    else:
        LOGGER.info(f"Use method {multivariate_method}")

    turns = ["word", "context"]
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    if lambda0 is not None:
        LOGGER.info(f"Use provided lambda0: {lambda0}")
    else:
        lambda0 = e.lambda0
        LOGGER.info(f"Use lambda0 from the embedding object: {lambda0}")

    logprobs = []
    LOGGER.info(f"Aggregate data...")
    X_cache, N_wd_cache, kappa_cache = aggregate_data(data, words, e)

    LOGGER.info(f"Split into independent sets...")
    blocks = split_into_independent_sets(e=e, wordcounts=N_wd_cache)

    # Pre-calculate quantities for the Laplacian prior
    edges, edgecounts = None, None
    if hasattr(e, "graph"):
        LOGGER.info(f"Calculate quantities for the Laplacian prior...")
        edges, edgecounts = [], []
        for wd in blocks:
            edges_wd = [list(e.graph.neighbors(wd_i)) for wd_i in wd]
            edges.append(edges_wd)
            edgecounts_wd = [len(e.graph.edges(wd_i)) for wd_i in wd]
            edgecounts.append(edgecounts_wd)
        edges = tf.ragged.constant(edges)
        edgecounts = tf.ragged.constant(edgecounts, dtype=tf.float64)

    for ix, turn in enumerate(turns * rounds):
        LOGGER.train(f"Flip turn: {turn}, {ix}")
        prior_count = 0

        # Calculate log_posterior and yield sample
        if ix % (yield_every * ll_every * 2) == 0:
            e_sample = copy.deepcopy(e)
            data_i = tf.constant([elem[0] for elem in data])
            data_j = tf.constant([elem[1] for elem in data])
            data_x = tf.constant([elem[2] for elem in data], dtype=tf.float64)

            LOGGER.info(f"Calculate log posterior for the sample...")
            ll, batch_size = 0.0, 10000
            if len(data_i) <= batch_size:
                LOGGER.info(f"Calculate log posterior for the whole data...")
                ll = tf.reduce_sum(sgns_likelihood(e, data_i, data_j, x=data_x))
            else:
                valid_batches = len(data_i) // batch_size
                LOGGER.info(f"Calculate log posterior for the sample in {valid_batches} batches...")
                for batch_ix in tqdm.tqdm(list(range(valid_batches))):
                    s_ix, e_ix = batch_ix * batch_size, (batch_ix +1) * batch_size
                    ll += tf.reduce_sum(sgns_likelihood(e, data_i[s_ix:e_ix], data_j[s_ix:e_ix], x=data_x[s_ix:e_ix]))

            posterior = ll + e.log_prob(len(data_i), len(data_i))
            logprobs.append(posterior)
            LOGGER.train(f"Log posterior for the sample: {posterior}")
            LOGGER.train(f"Avg log likelihood for the sample: {ll / len(data_x)}")
            yield e_sample

        # Plot log_posterior graph
        if ix % (yield_every * 100) == 0 and ix > 0 and plot:
            from matplotlib import pyplot as plt
            plot_x_range = np.array(range(len(logprobs)))
            plot_x_range = plot_x_range * ll_every
            plt.plot(plot_x_range, logprobs)
            plt.show()

        for block_ix, wd in progressbar.progressbar(enumerate(blocks)):
            e_theta = e.theta.numpy()
            if turn == "context":
                wd = [f"{wd_i}_c" for wd_i in wd]

            # Laplacian prior stuff
            mu_prior_wd_all, sigma_prior_wd_all = None, None
            if edgecounts is not None and turn != "context":
                sigma_prior_wd_all = get_laplacian_sigma(e, len(wd), tf.float64, edgecounts=edgecounts[block_ix])
                mu_prior_wd_all = get_laplacian_mu(e, edges=edges[block_ix], edgecounts=edgecounts[block_ix])
            else:
                sigma_prior_wd_all = get_laplacian_sigma(e, len(wd), tf.float64)

            # Some indices are skipped due to no data
            wd_nonskipped_indices = tf.squeeze(tf.where([sum(N_wd_cache[wd_i]) >= 1 for wd_i in wd]))
            wd_skipped_indices = tf.squeeze(tf.where([sum(N_wd_cache[wd_i]) < 1 for wd_i in wd]))
            wd_skipped = [wd_i for wd_i in wd if sum(N_wd_cache[wd_i]) < 1]
            wd = [wd_i for wd_i in wd if sum(N_wd_cache[wd_i]) >= 1]

            if len(wd) >= 1:
                # tf.squeeze flattens lists of length 1 to a scalar; revert that
                if len(wd) == 1:
                    wd_nonskipped_indices = tf.constant([wd_nonskipped_indices.numpy()])

                X_wd = [X_cache[wd_i] for wd_i in wd]
                X_wd = tf.ragged.constant(X_wd)
                X_padded = e[X_wd].to_tensor()

                K = e.dimensionality
                beta_init = tf.random.normal([len(wd), K], dtype=tf.float64) / K
                kappa_wd = tf.ragged.constant([kappa_cache.get(wd_i) for wd_i in wd])
                kappa_padded = kappa_wd.to_tensor()
                kappa_padded = tf.cast(kappa_padded, dtype=tf.float64)

                N_wd_ragged = tf.ragged.constant([N_wd_cache[wd_i] for wd_i in wd])
                N_wd_padded = N_wd_ragged.to_tensor()
                N_wd_padded = tf.math.maximum(N_wd_padded, tf.ones(N_wd_padded.shape, dtype=N_wd_padded.dtype))
                
                last_sample, mu_prior_wd = None, None
                sigma_prior_wd = tf.gather(sigma_prior_wd_all, wd_nonskipped_indices)
                if mu_prior_wd_all is not None:
                    mu_prior_wd = tf.gather(mu_prior_wd_all, wd_nonskipped_indices)

                betagen = polyagamma_sampler_tf(beta_init, X_padded,
                            None, kappa=kappa_padded, sigma_prior=sigma_prior_wd,
                            mu_prior=mu_prior_wd, N=N_wd_padded, iterations=polyagamma_iter,
                            multivariate_method=multivariate_method)
                for beta_sample in betagen:
                    last_sample = beta_sample

                wds_nonfreeze = [wd_i for wd_i in wd if wd_i not in freeze_params]
                ids_nonfreeze = [ix for ix, wd_i in enumerate(wd) if wd_i not in freeze_params]
                e[wds_nonfreeze] = tf.gather(beta_sample, ids_nonfreeze)

            # Sample from the prior
            if len(wd_skipped) > 0:
                # tf.squeeze flattens lists of length 1 to a scalar; revert that
                if len(wd_skipped) == 1:
                    wd_skipped_indices = tf.constant([wd_skipped_indices.numpy()])

                mu_skipped, sigma_skipped = None, tf.gather(sigma_prior_wd_all, wd_skipped_indices)
                if mu_prior_wd_all is not None:
                    mu_skipped = tf.gather(mu_prior_wd_all, wd_skipped_indices)

                for ix, wd_i in enumerate(wd_skipped):
                    if wd_i not in freeze_params:
                        mu_i, sigma_i = None, sigma_skipped[ix]
                        if mu_skipped is not None:
                            mu_i = mu_skipped[ix]
                        if turn == "context":
                            e[wd_i] = prior_sampler(e[wd_i], mu_prior=mu_i, sigma_prior=sigma_i)
                            prior_count += 1
                        else:
                            e[wd_i] = prior_sampler(e[wd_i], mu_prior=mu_i, sigma_prior=sigma_i)
                            prior_count += 1

        if prior_count >= len(words) * 0.2:
            LOGGER.warning(f"sampled from prior: {prior_count} out of {len(words)}")
        else:
            LOGGER.info(f"sampled from prior: {prior_count} out of {len(words)}")


def cbow_gibbs_parallellized(e, data, rounds=10, yield_every=1, freeze_params=[], plot=True, ll_every=1, batch_size=1):
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]

    lambda0 = e.lambda0
    LOGGER.info(f"Use lambda0 from the embedding object: {lambda0}")

    w = []
    C = []
    x = []

    K = e.dimensionality
    V = len(words)

    # For filtering out rho's that don't appear in the data
    rhos_in_data = set()
    alphas_in_data = set()

    alpha_co_occurences = nx.Graph()

    # Save the data indices of occurences of each rho and
    # alpha in the data
    rho_data_indices = [[] for _ in range(V)]
    alpha_data_indices = [[] for _ in range(V)]

    for i, elem in tqdm.tqdm(enumerate(data)):
        w_i, C_i, x_i = elem["w"], elem["C"], elem["x"]
        w.append(w_i)

        rhos_in_data.add(w_i)

        w_i_e_index = e.vocabulary[w_i]
        rho_data_indices[w_i_e_index] = rho_data_indices[w_i_e_index] + [i]

        for v in C_i:
            alphas_in_data.add(v)
            v_e_index = e.vocabulary[v]
            alpha_data_indices[v_e_index] = alpha_data_indices[v_e_index] + [i]

        for v1 in C_i:
            for v2 in C_i:
                if v1 != v2:
                    alpha_co_occurences.add_edge(v1, v2)

        C.append(C_i)
        x.append(x_i)

    w = tf.constant(w)
    C = tf.constant(C) + "_c"
    x = tf.constant(x)

    # Kappa is defined as a - b / 2; for the CBOW model
    # we always have a \in {0, 1}, and b = 1
    kappa = tf.cast(x, tf.float64) - 0.5

    rho_data_indices = tf.ragged.constant(rho_data_indices)
    alpha_data_indices = tf.ragged.constant(alpha_data_indices)

    LOGGER.debug(f"w shape {w.shape}")
    LOGGER.debug(f"C shape {C.shape}")
    LOGGER.debug(f"x shape {x.shape}")

    LOGGER.debug(f"w shape {w}")
    LOGGER.debug(f"C shape {C}")
    LOGGER.debug(f"x shape {x}")

    B_inv_rho = tf.eye(K, dtype=tf.float64) * e.lambda0
    B_inv_alpha = tf.eye(K, dtype=tf.float64) * e.lambda0

    for ix in range(rounds):
        LOGGER.train(f"Flip turn {ix}")
        prior_count = 0

        # Sample omega from the Polya-Gamma distribution
        rhos = e[w]
        alphas = tf.reduce_sum(e[C], axis=1)
        LOGGER.debug(f"rhos {rhos.shape}")
        LOGGER.debug(f"alphas {alphas.shape}")

        etas = tf.reduce_sum(alphas * rhos, axis=-1)

        # For the CBOW model, the b parameter is always 1
        # since we cannot do data aggregation
        omega = tf.constant(random_polyagamma(1, np.array(etas)))
        LOGGER.debug(f"omega {omega}")

        # Sample rho's given omega and alpha, a batch of batch_size at a time
        for j in range(V // batch_size):
            j0, j1 = batch_size * j, batch_size * (j+1)
            words_batch = words[j0:j1]

            # Only calculate A for the parameters that appear in the data
            words_with_data = [wd for wd in words_batch if wd in rhos_in_data]

            # Words without data are sampled directly from the prior
            words_without_data = [wd for wd in words_batch if wd not in rhos_in_data]

            if len(words_with_data) >= 1:
                wwd_len = len(words_with_data)
                LOGGER.debug(f"sample rhos: {words_with_data}")

                # Gather data
                indices_batch = e.tf_vocabulary[tf.constant(words_with_data)]
                occurences_batch = tf.gather(rho_data_indices, indices_batch)
                C_U = tf.gather(C, occurences_batch)

                # For sampling rhos, we can just sum the context vectors alpha
                # To get alpha^U
                A_U = e[C_U]
                A_U = tf.reduce_sum(A_U, axis=-2)

                # Fetch omegas for the correct indices
                omega_U = tf.gather(omega, occurences_batch)

                # expand last dim to make broadcasting possible
                omega_scaled_A_U = tf.expand_dims(omega_U, axis=-1) * A_U

                # calculate the precision matrices
                A_T_Omega_A =  tf.linalg.matmul(omega_scaled_A_U, A_U, transpose_a=True)
                A_T_Omega_A = A_T_Omega_A.to_tensor() # always of size (batch_size, K, K)

                # Add prior to get V_omega
                V_omega_rho_inv = A_T_Omega_A + B_inv_rho

                # Mean 
                kappa_U = tf.gather(kappa, occurences_batch)
                LOGGER.debug(f"kappa_U (rho) shape: {kappa_U.shape}, A_T_Omega_A: {A_T_Omega_A.shape}")
                LOGGER.debug(f"kappa_U: {type(kappa_U)}; A_T_Omega_A: {type(A_T_Omega_A)}")

                # TODO: do an efficient solve

                LOGGER.debug(f"A_U.shape: {A_U.shape}; kappa_U: {kappa_U.shape}")

                # Calculate the mean mu_omega
                #print(A_U.shape, kappa_U.shape)
                A_T_Kappa_U =  tf.linalg.matvec(A_U, kappa_U, transpose_a=True)
                # A_T_Kappa_U is always of size (batch_size, K)
                A_T_Kappa_U = A_T_Kappa_U.to_tensor()

                # Cholesky decompose the precision matrix
                #L_omega_rho_inv = tf.linalg.cholesky(V_omega_rho_inv)
                
                V_omega_rho = tf.linalg.inv(V_omega_rho_inv)
                L_omega_rho = tf.linalg.cholesky(V_omega_rho)

                # Dual Cholesky solve for mu_omega: 
                # LL^T mu = y
                # First: L b = y
                # Then: L^T mu = b
                #mu_omega_b = tf.linalg.triangular_solve(L_omega_rho_inv, tf.expand_dims(A_T_Kappa_U, axis=-1))
                #mu_omega = tf.linalg.triangular_solve(tf.transpose(L_omega_rho_inv, perm=[0,2,1]), mu_omega_b, lower=False)
                mu_omega = tf.linalg.matmul(V_omega_rho, tf.expand_dims(A_T_Kappa_U, axis=-1))

                y_delta = tf.random.normal((wwd_len, K, 1), dtype=tf.float64)
                x_delta = tf.linalg.matmul(V_omega_rho, y_delta)
                #x_delta = tf.linalg.triangular_solve(tf.transpose(L_omega_rho_inv, perm=[0,2,1]), y_delta, lower=False)

                # Single Cholesky solve for the offset
                #y_delta = tf.random.normal((wwd_len, K, 1), dtype=tf.float64)
                #x_delta = tf.linalg.triangular_solve(tf.transpose(L_omega_rho_inv, perm=[0,2,1]), y_delta, lower=False)

                new_vals = mu_omega + x_delta
                new_vals = tf.reduce_sum(new_vals, axis=-1)
                LOGGER.debug(f"{new_vals} {new_vals.shape}")

                e[words_with_data] = new_vals

            # Sample params without data from prior
            if len(words_without_data) >= 1:
                wwo_len = len(words_without_data)
                L_inv = tf.linalg.cholesky(B_inv_rho)
                
                x_omega = tf.linalg.triangular_solve(
                    tf.transpose(L_inv), tf.random.normal((K, wwo_len), dtype=tf.float64),
                    lower=False
                )

                e[words_without_data] = tf.transpose(x_omega)

        etas = tf.reduce_sum(alphas * rhos, axis=-1)
        deltas = etas

        # TODO: sample alphas
        for j in range(V // batch_size):
            j0, j1 = batch_size * j, batch_size * (j+1)
            words_batch = words[j0:j1]

            # Only calculate R for the parameters that appear in the data
            words_with_data = [wd for wd in words_batch if wd in alphas_in_data]

            # Words without data are sampled directly from the prior
            words_without_data = [wd for wd in words_batch if wd not in alphas_in_data]

            words_with_data_c = [f"{wd}_c" for wd in words_with_data]
            words_without_data_c = [f"{wd}_c" for wd in words_without_data]

            # Sample params with data from posterior 
            if len(words_with_data) >= 1:
                wwd_len = len(words_with_data)
                co_occurences_U = alpha_co_occurences.subgraph(words_with_data)

                V_omega_inv = tf.zeros((K * wwd_len, K * wwd_len), dtype=tf.float64)
                LOGGER.debug(f"sample alphas: {words_with_data}")

                indices_batch = e.tf_vocabulary[tf.constant(words_with_data)]
                occurences_batch = tf.gather(alpha_data_indices, indices_batch)

                w_U = tf.gather(w, occurences_batch)
                R_U = e[w_U]

                # Fetch omegas for the correct indices
                omega_U = tf.gather(omega, occurences_batch)

                # expand last dim to make broadcasting possible
                R_U_omega_U = tf.expand_dims(omega_U, axis=-1) * R_U

                # calculate the precision matrices
                R_T_Omega_R =  tf.linalg.matmul(R_U_omega_U, R_U, transpose_a=True)
                # R_T_Omega_R is always of size (batch_size, K, K)
                R_T_Omega_R = R_T_Omega_R.to_tensor()

                # Initialize V_omega as a block matrix consisting of the prior
                V_Omega = tf.experimental.numpy.kron(tf.eye(wwd_len), B_inv_alpha)

                # Add R_T_Omega_R as block diagonals
                for j in range(wwd_len):
                    # Create tensor with one nonzero element at [j,j]
                    E_diag = tf.sparse.to_dense(tf.sparse.SparseTensor([[j,j]], [1.0], [wwd_len, wwd_len]))
                    V_Omega += tf.experimental.numpy.kron(E_diag, R_T_Omega_R[j])

                #print_tf_tensor(V_Omega, 3)

                # TODO: deal with the off-diagonal
                for ix, wds in enumerate(co_occurences_U.edges):
                    j, k = wds
                    indices_j = e.tf_vocabulary[tf.constant([j])]
                    indices_k = e.tf_vocabulary[tf.constant([k])]

                    occurences_j = tf.gather(alpha_data_indices, indices_j).to_tensor()
                    occurences_k = tf.gather(alpha_data_indices, indices_k).to_tensor()

                    occurences_jk = tf.sparse.to_dense(tf.sets.intersection(occurences_j, occurences_k))

                    w_jk = tf.gather(w, occurences_jk)
                    omega_jk = tf.gather(omega, occurences_jk)
                    R_jk = e[w_jk]

                    # R_T_Omega_R is always of size (batch_size, K, K)
                    R_jk_omega_jk = tf.expand_dims(omega_jk, axis=-1) * R_jk
                    R_jk_omega_jk_R_jk =  tf.linalg.matmul(R_jk_omega_jk, R_jk, transpose_a=True)
                    assert R_jk_omega_jk_R_jk.shape[0] == 1

                    R_jk_omega_jk_R_jk = tf.reduce_sum(R_jk_omega_jk_R_jk, axis=0)
                    #exit()

                    # Create tensor with two nonzero elements at [j,k] and [k, j]
                    j = words_with_data.index(j)
                    k = words_with_data.index(k)
                    E_sparsetensor = tf.sparse.SparseTensor([[j,k], [k,j]], [1.0, 1.0], [wwd_len, wwd_len])
                    E_diag = tf.sparse.to_dense(tf.sparse.reorder(E_sparsetensor))
                    V_Omega += tf.experimental.numpy.kron(E_diag, R_jk_omega_jk_R_jk)

                #L_Omega = tf.linalg.cholesky(V_Omega)

                # TODO: properly calculate inverse
                V_Omega_inv = tf.linalg.inv(V_Omega)
                L_Omega_inv = tf.linalg.cholesky(V_Omega_inv)

                # TODO: deal with delta etc.

                # Steps
                # 0. precalculated previous eta
                # 1a. Gather the indices to be modified (U) as a ragged tensor
                # 1b. Gather rhos for the indices
                # 2. Calculate alpha^T rho for each of the indices to be modified
                # 3. Flatten ragged tensors via rt.flat_values
                # 4. Use tf.scatter_nd to get the contribution of U
                # 5. Calculate delta := eta - scattered_update
                # 6. Gather delta_U as a ragged tensor
                # 7. (pre)-calculate next eta

                alpha_U = e[words_with_data]
                alpha_U = tf.expand_dims(alpha_U, axis=1)
                delta_U = R_U * alpha_U
                delta_U = tf.reduce_sum(delta_U, axis=-1)

                # Flatten arrays for nd_scatter
                delta_U_flat = delta_U.flat_values
                occurences_U_flat = occurences_batch.flat_values
                
                # Re-calculate etas
                rhos = e[w]
                alphas = tf.reduce_sum(e[C], axis=1)
                etas = tf.reduce_sum(alphas * rhos, axis=-1)

                # Addition to eta contributes by U
                addition = tf.scatter_nd(tf.expand_dims(occurences_U_flat, axis=-1), delta_U_flat, etas.shape)
                deltas = etas - addition

                delta_U = tf.gather(deltas, occurences_batch)
                delta_U_omega_U = delta_U * omega_U

                kappa_minus_delta_omega = tf.gather(kappa, occurences_batch) - delta_U_omega_U

                # Find mu_omega via Cholesky: double tridiagonal solve
                R_U_kdo = tf.linalg.matvec(R_U, kappa_minus_delta_omega, transpose_a=True).flat_values
                #mu_omega_inv = mu_omega_inv.flat_values
                #mu_omega_inv1 = tf.linalg.triangular_solve(
                #    tf.transpose(L_Omega), tf.expand_dims(mu_omega_inv, axis=-1),
                #    lower=False
                #)
                #mu_omega = tf.linalg.triangular_solve(L_Omega, mu_omega_inv1)

                #print(V_Omega_inv.shape, R_U_kdo.shape)
                #print(R_U_kdo)
                #print(V_Omega_inv)
                mu_omega = tf.linalg.matvec(V_Omega_inv, R_U_kdo)
                x_omega = tf.linalg.matvec(L_Omega_inv, tf.random.normal((wwd_len * K,), dtype=tf.float64))

                # Add covariance noise via Cholesky: single tridiagonal solve
                #x_omega = tf.linalg.triangular_solve(
                #    tf.transpose(L_Omega), tf.random.normal((wwd_len * K, 1), dtype=tf.float64),
                #    lower=False
                #)
                new_vals = mu_omega + x_omega

                first_vec = new_vals[:K]
                new_vals = tf.reshape(new_vals, [wwd_len, K])
                first_vec_prime = new_vals[0]

                #print(first_vec, first_vec_prime)
                assert_msg = f"First element of reshaped array {first_vec_prime} should be same as the K first entries {first_vec}"
                assert first_vec_prime.shape == first_vec.shape, f"Vector shapes should match {first_vec_prime.shape} vs {first_vec.shape}"
                assert np.max(np.abs(first_vec_prime - first_vec)) < 0.00001, assert_msg

                e[words_with_data_c] = new_vals

            # Sample params without data from prior
            if len(words_without_data) >= 1:
                wwo_len = len(words_without_data)
                L_inv = tf.linalg.cholesky(B_inv_alpha)
                
                x_omega = tf.linalg.triangular_solve(
                    tf.transpose(L_inv), tf.random.normal((K, wwo_len), dtype=tf.float64),
                    lower=False
                )

                e[words_without_data_c] = tf.transpose(x_omega)

        sigm = tf.math.sigmoid(etas)
        x64 = tf.cast(x, dtype=tf.float64)
        p = sigm * x64  + (1-x64) * (1-sigm)
        log_p = tf.math.log(p)
        log_ll = tf.reduce_mean(log_p)
        log_posterior = log_ll + e.log_prob(batch_size=1, data_size=len(x))

        print("log_ll", log_ll.numpy(), "log_posterior", log_posterior.numpy())

        yield copy.deepcopy(e)







