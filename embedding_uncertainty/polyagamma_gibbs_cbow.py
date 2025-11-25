from trainerlog import get_logger
LOGGER = get_logger("sampling", splitsec=True)
LOGGER.info("Load modules..")
import numpy as np
from polyagamma import random_polyagamma
import copy
import tensorflow as tf
import networkx as nx
import tqdm

LOGGER.info("Done!")
from time import perf_counter as pc

def cbow_gibbs_sampler(w_idx, C_idx, x_vec,
                    *, n_samples, S, V, K, lam=None):
    """
    S : number of inner samples
    """

    if lam is None:
        lam = K
        
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
                    Vw = np.linalg.inv(AOA + lam * np.eye(K))
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

def sample_cbow_omegas(e, w, C):
    rhos = e[w]
    alphas = tf.reduce_sum(e[C], axis=1)
    etas = tf.reduce_sum(alphas * rhos, axis=-1)
    omega = tf.constant(random_polyagamma(1, np.array(etas)))
    return omega

def cbow_gibbs_parallellized(e, data, rounds=10, yield_every=1, freeze_params=[], plot=True, ll_every=1, batch_size=10):
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]

    lambda0 = e.lambda0
    LOGGER.info(f"K: {e.dimensionality}")
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

    B_inv_rho = tf.eye(K, dtype=tf.float64) * e.lambda0
    B_inv_alpha = tf.eye(K, dtype=tf.float64) * e.lambda0

    for ix in range(rounds):
        LOGGER.train(f"Flip turn {ix}")
        prior_count = 0

        # Sample omega from the Polya-Gamma distribution
        LOGGER.debug(f"Calculate eta in order to sample omega")

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

        ### RHO ###
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

                LOGGER.debug(f"Calculate empirical precision matrix A_T_O_A: omega scaling")
                # expand last dim to make broadcasting possible
                omega_scaled_A_U = tf.expand_dims(omega_U, axis=-1) * A_U

                # calculate the precision matrices
                LOGGER.debug(f"Calculate empirical precision matrix A_T_O_A: matrix product")
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

                LOGGER.debug(f"Invert V_omega_rho_inv")
                V_omega_rho = tf.linalg.inv(V_omega_rho_inv)

                LOGGER.debug(f"Cholesky decompose V_omega_rho")
                L_omega_rho = tf.linalg.cholesky(V_omega_rho)

                # TODO: Dual Cholesky solve for mu_omega: 
                # LL^T mu = y
                # First: L b = y
                # Then: L^T mu = b
                LOGGER.debug(f"Calculate mean for rho_U")
                mu_omega = tf.linalg.matvec(V_omega_rho, A_T_Kappa_U)

                LOGGER.debug(f"Sample via the Cholesky factor")
                y_delta = tf.random.normal((wwd_len, K), dtype=tf.float64)
                x_delta = tf.linalg.matvec(L_omega_rho, y_delta)

                # TODO: Single Cholesky solve for the offset
                new_vals = mu_omega + x_delta
                LOGGER.debug(f"New vals {new_vals.shape}")

                e[words_with_data] = new_vals

            # Sample params without data from prior
            if len(words_without_data) >= 1:
                # TODO: pre-calculate prior Cholesky
                wwo_len = len(words_without_data)

                LOGGER.debug(f"Invert prior")
                V_inv = tf.linalg.inv(B_inv_rho)

                LOGGER.debug(f"Cholesky decompose prior")
                L_inv = tf.linalg.cholesky(V_inv)
                x_noise = tf.random.normal((K, wwo_len), dtype=tf.float64)
                e[words_without_data] = tf.matvec(L_inv, x_noise)

        LOGGER.debug(f"Sample omegas")
        omega = sample_cbow_omegas(e, w, C)

        # ALPHA
        LOGGER.debug(f"Sample alphas")
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

                LOGGER.debug(f"sample alphas for: {words_with_data}")

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

                # off-diagonal
                LOGGER.debug(f"Calculate the off-diagonal")
                time_off_diag_ROR = 0.0
                time_off_diag_kronecker = 0.0
                for ix, wds in enumerate(co_occurences_U.edges):
                    j, k = wds
                    
                    t0 = pc()

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

                    # Create tensor with two nonzero elements at [j,k] and [k, j]
                    t1 = pc()
                    #LOGGER.debug(f"Add result to V_omega via the Kronecker product")
                    j = words_with_data.index(j)
                    k = words_with_data.index(k)
                    E_sparsetensor = tf.sparse.SparseTensor([[j,k], [k,j]], [1.0, 1.0], [wwd_len, wwd_len])
                    E_diag = tf.sparse.to_dense(tf.sparse.reorder(E_sparsetensor))
                    V_Omega += tf.experimental.numpy.kron(E_diag, R_jk_omega_jk_R_jk)

                    t2 = pc()
                    time_off_diag_ROR += t1 - t0
                    time_off_diag_kronecker += t2 - t1

                LOGGER.debug(f"Time spent calculating ROR: {time_off_diag_ROR}")
                LOGGER.debug(f"Time spent in the Kronecker product: {time_off_diag_kronecker}")

                #L_Omega = tf.linalg.cholesky(V_Omega)
                #exit()
                # TODO: properly calculate inverse
                LOGGER.debug(f"Invert V_Omega_inv")
                V_Omega_inv = tf.linalg.inv(V_Omega)
                LOGGER.debug(f"Cholesky factorize V_Omega")
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

                alpha_U = e[words_with_data_c]
                addition_U = tf.linalg.matvec(R_U, alpha_U)
                addition_U_flat = addition_U.flat_values
                occurences_U_flat = occurences_batch.flat_values
                
                # Re-calculate etas
                # TODO: only do for the batch
                LOGGER.debug(f"Calculate eta")
                rhos = e[w]
                alphas = tf.reduce_sum(e[C], axis=1)
                etas = tf.reduce_sum(alphas * rhos, axis=-1)

                # Addition to eta contributes by U
                LOGGER.debug(f"Calculate delta")
                addition = tf.scatter_nd(tf.expand_dims(occurences_U_flat, axis=-1), addition_U_flat, etas.shape)
                deltas = etas - addition

                LOGGER.debug(f"Calculate mean for alpha_U")
                delta_U = tf.gather(deltas, occurences_batch)
                delta_U_omega_U = delta_U * omega_U

                kappa_U = tf.gather(kappa, occurences_batch)
                kappa_minus_delta_omega = kappa_U - delta_U_omega_U

                R_U_kdo = tf.linalg.matvec(R_U, kappa_minus_delta_omega, transpose_a=True).flat_values

                # TODO: Find mu_omega and random offset via Cholesky and vector solve
                mu_omega = tf.linalg.matvec(V_Omega_inv, R_U_kdo)
                x_omega = tf.linalg.matvec(L_Omega_inv, tf.random.normal((wwd_len * K,), dtype=tf.float64))

                # Reshape new vals to a matrix
                new_vals = mu_omega + x_omega
                first_vec = new_vals[:K]
                new_vals = tf.reshape(new_vals, [wwd_len, K])
                first_vec_prime = new_vals[0]

                # Ensure that reshaping worked
                assert_msg = f"First element of reshaped array {first_vec_prime} should be same as the K first entries {first_vec}"
                assert first_vec_prime.shape == first_vec.shape, f"Vector shapes should match {first_vec_prime.shape} vs {first_vec.shape}"
                assert np.max(np.abs(first_vec_prime - first_vec)) < 0.00001, assert_msg

                e[words_with_data_c] = new_vals

            # Sample params without data from prior
            if len(words_without_data) >= 1:
                LOGGER.debug(f"Sample prior for alpha_U without data")
                wwo_len = len(words_without_data)
                V_inv = tf.linalg.inv(B_inv_alpha)
                L_inv = tf.linalg.cholesky(V_inv)
                
                x_noise = tf.random.normal((K, wwo_len), dtype=tf.float64)
                e[words_without_data_c] = tf.matvec(L_inv, x_noise)
        
        # Calculate likelihood
        LOGGER.debug(f"Calculate likelihood")
        rhos = e[w]
        alphas = tf.reduce_sum(e[C], axis=1)
        etas = tf.reduce_sum(alphas * rhos, axis=-1)

        x64 = tf.cast(x, dtype=tf.float64)
        sigm = tf.math.sigmoid(etas * (2.0 * x64 - 1.0))
        log_p = tf.math.log(sigm)
        log_ll = tf.reduce_mean(log_p)
        log_posterior = log_ll + e.log_prob(batch_size=1, data_size=len(x))

        LOGGER.train(f"log_ll {log_ll.numpy()}, log_posterior {log_posterior.numpy()}")
        LOGGER.train(f"eta {etas.numpy()[:3]}")

        yield copy.deepcopy(e)







