import tensorflow as tf
import numpy as np
import copy
import networkx as nx
import progressbar as pb

from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.models import cbow_likelihood
from probabilistic_word_embeddings.estimation import map_estimate
from pathlib import Path

from numpy import dot
from numpy.linalg import norm

def cossim(a, b):
    return dot(a,b) / (norm(a) * norm(b))

def gradient_descent(e, data, steps, epsilon=0.005):
    i, c_j, x = data
    x_prime = tf.cast(x * 2.0 - 1.0, tf.float64)
    print(x_prime)
    wds = [str(wd) for wd in list(set(i))]
    print("wds:", wds)
    e_gradient = copy.deepcopy(e)
    #e[wds] = tf.random.normal(e[wds].shape, dtype=tf.float64) * 0.0001

    for step in pb.progressbar(range(steps)):
        with tf.GradientTape() as tape:
            eta = tf.multiply(c_j, e[i])
            eta = tf.reduce_sum(eta, axis=1)
            eta = tf.multiply(eta, x_prime)
            print(eta)
            log_prob = tf.reduce_sum(tf.math.log(tf.math.sigmoid(eta))) - 0.5 * tf.reduce_sum(tf.multiply(e[wds], e[wds]))
            print(log_prob)
            gradient = tape.gradient(log_prob, e.theta)
            e_gradient.theta.assign(gradient)
        
        e[wds] = e[wds] + e_gradient[wds] * epsilon
    return e

def conditional_maximize(e, steps=100, epsilon=0.01):
    e = copy.deepcopy(e)
    p = Path("data/stan")
    c_j = np.load(p / "e_j.npy")
    c_j_neg = np.load(p / "e_j_neg.npy")
    i = np.load(p / "i.npy")

    print(c_j)
    print(c_j_neg)

    print(c_j.shape)
    print(c_j_neg.shape)

    x_1 = tf.ones(c_j.shape[0])
    x_0 = tf.zeros(c_j_neg.shape[0])
    x = tf.concat([x_1, x_0], axis=0)
    c_j = tf.concat([c_j, c_j_neg], axis=0)
    i = np.full(c_j.shape[0], i[0])
    print(i)

    print(c_j.shape)
    print(x.shape)

    data = i, c_j, x
    e = gradient_descent(e, data, steps, epsilon=epsilon)
    return e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()
    
    e = Embedding(saved_model_path="./lapl_emb.pkl")
    print(cossim(e["dog"], e["dog"]))
    print(cossim(e["dog"], e["dogs"]))
    print(cossim(e["dog"], e["cat"]))
    print(cossim(e["dog"], e["animal"]))
    print(cossim(e["dog"], e["man"]))
    print(cossim(e["dog"], e["tree"]))
    print(cossim(e["dog"], e["missing"]))
    print(cossim(e["dog"], e["font"]))
    print(cossim(e["dog"], e["remove"]))
    e_prime = conditional_maximize(e, steps=args.steps, epsilon=args.epsilon)
    print(e["dog"])
    print(e_prime["dog"])

    print(cossim(e["dog"], e_prime["dog"]))


