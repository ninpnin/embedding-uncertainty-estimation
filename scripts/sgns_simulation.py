import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy
import networkx as nx
import progressbar as pb
import json

from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.models import cbow_likelihood
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.utils import transfer_embeddings
import random

def generate_dataset(e, batches, batch_size, seed=None):
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]

    for batch in range(batches):
        i = random.choices(words, k=batch_size)
        j = random.choices(words, k=batch_size)
        j = [wd + "_c" for wd in j]
        print(i)
        print(j)

        e_i, e_j = e[i], e[j]

        etas = tf.math.sigmoid(tf.reduce_sum(tf.multiply(e_i, e_j), axis=1))
        print(etas)
        x = tfp.distributions.Bernoulli(probs=etas).sample()
        print(x)

        i_j_x = (i,j,x)
        yield i_j_x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--words", type=str, nargs="+", default=["this", "will", "about", "food", "democrats", "republicans"])
    parser.add_argument("--modelpath", type=str, default="trained/lapl_emb.pkl")
    parser.add_argument("--dim", type=int, default=100)
    args = parser.parse_args()
    
    e = Embedding(saved_model_path=args.modelpath)
    words = args.words
    e_prime = Embedding(set(words), dimensionality=args.dim)
    e_prime = transfer_embeddings(e, e_prime)

    batch_size = 10
    batches = args.N // batch_size
    d = {"i": [], "j": [], "x": []}
    for ix, data in enumerate(generate_dataset(e_prime, batches, batch_size)):
        print(ix)
        i, j, x = data
        print(i, j, x)

        d["i"] = d["i"] + i
        d["j"] = d["j"] + j
        d["x"] = d["x"] + [int(x_i) for x_i in x.numpy()]

    print(d)

    with open(f'data/sgns_simulated-dim-{args.dim}.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


