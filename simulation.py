import tensorflow as tf
import numpy as np
import copy
import networkx as nx
import progressbar as pb

from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.models import cbow_likelihood
from probabilistic_word_embeddings.estimation import map_estimate

from numpy import dot

def mean_and_cov(e, data, sample_size=2000, ws=5):
    data = tf.constant([wd for wd in data if wd in e])
    vocab = [wd for wd in e.vocabulary if "_c" not in wd]
    #rho = e[vocab[:sample_size]]
    #rho_mean = tf.reduce_mean(rho, axis=0)

    i = np.random.randint(len(data), size=sample_size)
    js1 = tf.ragged.range(i + 1, i + ws + 1)
    js2 = tf.ragged.range(i - ws, i )

    targets = tf.gather(data, i)
    contexts1 = tf.gather(data, js1) + "_c"
    contexts2 = tf.gather(data, js2) + "_c"

    rho = e[targets]
    alpha = tf.reduce_sum(e[contexts1], axis=1) + tf.reduce_sum(e[contexts2], axis=1)

    rho_mean = tf.reduce_mean(rho, axis=0)
    alpha_mean = tf.reduce_mean(alpha, axis=0)

    rho_cov = np.cov(tf.transpose(rho))
    alpha_cov = np.cov(tf.transpose(alpha))
    print(rho_cov.shape)
    print(alpha_cov.shape)

    return rho_mean, alpha_mean, rho_cov, alpha_cov


def random_eta(rho_mean, alpha_mean, rho_cov, alpha_cov):
    r = np.random.multivariate_normal(rho_mean, rho_cov)
    a = np.random.multivariate_normal(alpha_mean, alpha_cov)
    return dot(r, a)

def generate_embedding(rho_mean, alpha_mean, rho_cov, alpha_cov, words, ws=5):
    dim = rho_mean.shape[0]
    e = Embedding(set(words), dimensionality=dim)

    for wd in pb.progressbar(words):
        ws_prime = (ws * 2)
        r = np.random.multivariate_normal(rho_mean, rho_cov)

        # Ensure that the sum of WS alphas is going to have the correct mean/cov
        a = np.random.multivariate_normal(alpha_mean / ws_prime, alpha_cov / ws_prime)
        e[wd] = r
        e[wd + "_c"] = a

    return e

def generate_dataset(e, ws=5, size=20000):
    vocab = [wd for wd in e.vocabulary if "_c" not in wd]
    i = []
    j = []
    x = []

    for _ in pb.progressbar(list(range(size))):
        i_sample = tf.constant(np.random.choice(vocab))
        j_sample = tf.constant(np.random.choice(vocab, size=ws*2))
        j_sample = j_sample + "_c"
        alpha = tf.reduce_sum(e[j_sample], axis=0)
        rho = e[i_sample]

        eta = tf.sigmoid(dot(alpha, rho))
        print(eta)

        x_sample = 0.0
        if np.random.rand() <= eta:
            x_sample = 1.0

        i.append(str(i_sample))
        j.append(j_sample)
        x.append(x_sample)

    i = tf.constant(i)
    j = tf.stack(j, axis=1)
    x = tf.constant(x)
    print("x mean", tf.reduce_mean(x))
    return i, j, x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    text = open("wikismall.txt").read().lower().split()
    text, vocabulary = preprocess_standard(text)

    e = Embedding(saved_model_path="./lapl_emb.pkl")
    # Perform MAP estimation
    rho_mean, alpha_mean, rho_cov, alpha_cov = mean_and_cov(e, text)
    
    print(rho_mean)
    print(alpha_mean)
    print(dot(rho_mean, alpha_mean))
    print(tf.sigmoid(dot(rho_mean, alpha_mean)))

    etas = []
    probs = []

    for _ in pb.progressbar(range(100)):
        eta = random_eta(rho_mean, alpha_mean, rho_cov, alpha_cov)
        prob = tf.sigmoid(eta)
        etas.append(eta)
        probs.append(prob)

    print(np.mean(probs))
    print(np.array(etas))

    vocab_prime = list(set(text[:250]))
    e_prime = generate_embedding(rho_mean, alpha_mean, rho_cov, alpha_cov, vocab_prime)
    e_prime.save("e_sim.pkl")

    i, j, x = generate_dataset(e_prime, ws=5, size=20000)

    print(i)
    print(j)
    print(x)

