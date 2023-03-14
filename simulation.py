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
    print(dot(rho_mean, alpha_mean))
    print(tf.sigmoid(dot(rho_mean, alpha_mean)))

    etas = []
    probs = []

    for _ in pb.progressbar(range(1000)):
        eta = random_eta(rho_mean, alpha_mean, rho_cov, alpha_cov)
        prob = tf.sigmoid(eta)
        etas.append(eta)
        probs.append(prob)

    print(np.mean(probs))
    print(np.array(etas))

