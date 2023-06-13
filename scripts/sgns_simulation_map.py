import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy
import networkx as nx
import progressbar as pb
import json
from numpy import dot

from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.models import cbow_likelihood, sgns_likelihood
#from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.utils import transfer_embeddings
import random

def get_dataset(path):
    with open(path) as f:
        d = json.load(f)

    return d["i"], d["j"], d["x"]

def populate_emb(e, words, dim, stdev=1):
    e_prime[words] = np.random.randn(len(words), args.dim) * stdev / np.sqrt(dim)
    e_prime[[f"{w}_c" for w in words]] = np.random.randn(len(words), dim) * stdev / np.sqrt(dim)
    return e_prime

def map_estimate(e, i, j, x, batch_size=100, epochs=5):
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    N = len(i)
    for epoch in range(epochs):
        epoch_training_loss = []
        for b in range(0, len(i), batch_size):
            i_b, j_b, x_b = i[b:b + batch_size], j[b:b + batch_size], x[b:b + batch_size]
            i_b, j_b, x_b = tf.constant(i_b), tf.constant(j_b), tf.constant(x_b, dtype=tf.float64)
            #print(i_b, j_b, x_b)
            objective = lambda: - tf.reduce_sum(sgns_likelihood(e, i_b, j_b, x=x_b)) - e.log_prob(batch_size, N)
            _ = opt.minimize(objective, [e.theta])
            epoch_training_loss.append(objective() / len(i_b))
        print(f"Epoch {epoch} mean training loss after {b} batches: {np.mean(epoch_training_loss)}")

    return e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--modelpath", type=str, default=None)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    i, j, x = get_dataset(args.datapath)
    words = set(i)
    print(f"Vocabulary size {len(words)}")
    e_prime = Embedding(words, dimensionality=args.dim)
    e_prime = map_estimate(e_prime, i, j, x, batch_size=args.batch_size, epochs=args.epochs)
    print(e_prime.theta)

    if args.modelpath is not None:
        e = Embedding(saved_model_path=args.modelpath)
        print(e.theta)

        words = list(e.vocabulary)[:5]
        words_prime = list(e.vocabulary)[-5:]

        for w1, w2 in zip(words, words_prime):
            d1 = dot(e[w1], e[w2])
            d2 = dot(e_prime[w1], e_prime[w2])

            print(f"{w1} {w2}")
            print(f"Dot prod 1 {d1}")
            print(f"Dot prod 2 {d2}")






