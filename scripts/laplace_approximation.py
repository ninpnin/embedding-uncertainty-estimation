from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import copy
import json
import progressbar
import pandas as pd
from trainerlog import get_logger

LOGGER = get_logger("laplace")

def cossim(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_subhessian(alpha, rho, x=1):
    if x == 0:
        x = -1
    alpha = tf.Variable(alpha)
    rho = tf.Variable(rho)
    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        eta = tf.reduce_sum(x * alpha * rho)
        p = tf.math.sigmoid(eta)
        log_p = tf.math.log(p)

      g = t1.gradient(log_p, alpha)
    return t2.jacobian(g, rho)

def get_diag_subhessian(alpha, rho, x=1):
    if x == 0:
        x = -1
    alpha = tf.Variable(alpha)
    rho = tf.Variable(rho)
    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        eta = tf.reduce_sum(x * alpha * rho)
        p = tf.math.sigmoid(eta)
        log_p = tf.math.log(p)

      g = t1.gradient(log_p, alpha)
    return t2.jacobian(g, alpha)


def get_laplace_hessian_optimized(e, data):
    vocabulary = e.vocabulary
    lambda0 = e.lambda0
    dim = e.dimensionality
    hessian_vocabulary = {key: val * dim for key, val in vocabulary.items()}

    data_prime = {}
    for elem in data:
        data_prime[elem] = data_prime.get(elem, 0) + 1

    H = np.zeros((dim * len(vocabulary), dim * len(vocabulary)))

    # Spherical prior
    ## lambda0 * ( | alpha |^2 + | rho |^2 )
    H += - np.identity(dim * len(vocabulary)) * lambda0 * 2.0

    # Likelihood
    LOGGER.train(f"Calculate likelihood...")
    for elem, count in progressbar.progressbar(data_prime.items()):
        LOGGER.debug(f"{elem}, {count}")
        w, v, x = elem
        rho = e[w]
        alpha = e[v]
        subhessian = get_subhessian(alpha, rho, x=x)
        diag_subhessian_a = get_diag_subhessian(alpha, rho, x=x)
        diag_subhessian_r = get_diag_subhessian(rho, alpha, x=x)
        
        i, j = hessian_vocabulary[w], hessian_vocabulary[v]
        # Off-diagonal subhessian (symmetric)
        for m in range(0, dim):
            for n in range(0, dim):
                entry = subhessian[m,n]
                H[i + m, j + n] += entry * count
                H[j + n, i + m] += entry * count

        # Diagonal subhessian for rho
        for m in range(0, dim):
            for n in range(0, dim):
                entry = diag_subhessian_r[m,n]
                H[i + m, i + n] += entry * count

        # Diagonal subhessian for alpha
        for m in range(0, dim):
            for n in range(0, dim):
                entry = diag_subhessian_a[m,n]
                H[j + m, j + n] += entry * count

    return H, hessian_vocabulary

def sample(e, L, hessian_vocabulary):
    epsilon_prime = np.random.randn(L.shape[0])
    epsilon = L @ epsilon_prime
    dim = e.dimensionality

    e_sample = copy.deepcopy(e)

    for v in e.vocabulary:
        i = hessian_vocabulary[v]
        e_sample[v] += epsilon[i: i+dim]
    
    return e_sample

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, default="example_embedding.pkl")
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--word", type=str, default="word1")
    parser.add_argument("--context", type=str, default="word2_c")
    args = parser.parse_args()

    LOGGER.train(f"Load embedding from {args.embedding}...")
    e = Embedding(saved_model_path=args.embedding)
    sim = tf.math.sigmoid(tf.reduce_sum(e[args.word]* e[args.context]))
    LOGGER.debug(f"Sim (MAP) [{args.word} {args.context}] {sim}")
    e.lambda0 = 1.0
    LOGGER.debug(f"lambda0: {e.lambda0}")
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]
    if args.datapath is not None:
        with open(args.datapath, "rb") as f:
            d = json.load(f)

        LOGGER.debug(f"Keys: {d.keys()}")

        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            data.append((w,v,x))

    if args.data_len is not None:
        data = data[:data_len]
    #data = data[:20]
    #H, hessian_vocabulary = get_laplace_hessian(e, data)
    H, hessian_vocabulary = get_laplace_hessian_optimized(e, data)

    #assert np.max(np.abs(H- H_prime)) < 0.001,np.max(np.abs(H- H_prime))
    Sigma = np.linalg.inv(H)

    LOGGER.info("H")
    LOGGER.info(H)
    LOGGER.info("Sigma")
    LOGGER.info(Sigma)

    #words = [wd for wd in list(e.vocabulary) if "_c" not in wd]
    #contexts = [wd for wd in list(e.vocabulary) if "_c" in wd]
    words = [args.word]
    contexts = [args.context]
    rows = []
    for w in words:
        for v in contexts:
            LOGGER.info(f"Word pair: {w}, {v}")
            if w not in v:
                i, j = hessian_vocabulary[w], hessian_vocabulary[v]
                Sigma_11 = Sigma[i:i+e.dimensionality, i:i+e.dimensionality]
                Sigma_12 = Sigma[i:i+e.dimensionality, j:j+e.dimensionality]
                Sigma_21 = Sigma[j:j+e.dimensionality, i:i+e.dimensionality]
                Sigma_22 = Sigma[j:j+e.dimensionality, j:j+e.dimensionality]

                Sigma_prime = np.vstack([np.hstack([Sigma_11, Sigma_12]), np.hstack([Sigma_21, Sigma_22])])
                #print(Sigma_prime)
                LOGGER.info("Covariance:")
                LOGGER.info(np.array_str(Sigma_prime, precision=6, suppress_small=True))
                eigenval, _ = np.linalg.eig(Sigma_prime)
                #print(eigenval)
                L = np.linalg.cholesky(-Sigma_prime)
                #print(L)

                for s in range(100):
                    epsilon_prime = np.random.randn(L.shape[0])
                    epsilon = L @ epsilon_prime

                    rho_w = e[w] + epsilon[:e.dimensionality]
                    alpha_v = e[v] + epsilon[e.dimensionality:]

                    sim = tf.math.sigmoid(tf.reduce_sum(rho_w * alpha_v)).numpy()
                    LOGGER.debug(f"Sim {sim}")
                    rows.append([w,v, sim])
            else:
                LOGGER.debug("Skip")

        sim = cossim(e[w], e[v])
        LOGGER.info(f"Sim (MAP) {sim}")

        df = pd.DataFrame(rows, columns=["w", "v", "p_hat"])
        LOGGER.info(df)

        q05 = df["p_hat"].quantile(0.05)
        q95 = df["p_hat"].quantile(0.95)
        

        LOGGER.info(f"q05 {q05}")
        LOGGER.info(f"q95 {q95}")
        post_mean = df["p_hat"].mean()
        LOGGER.info(f"mean {post_mean}")
    #print(Sigma_12)

    #L = np.linalg.cholesky(Sigma_12)    


