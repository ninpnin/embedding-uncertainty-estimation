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

    for _ in pb.progressbar(range(batches)):
        i = random.choices(words, k=batch_size)
        j = random.choices(words, k=batch_size)
        j = [wd + "_c" for wd in j]

        e_i, e_j = e[i], e[j]
        etas = tf.math.sigmoid(tf.reduce_sum(tf.multiply(e_i, e_j), axis=1))
        x = tfp.distributions.Bernoulli(probs=etas).sample()

        i_j_x = (i,j,x)
        yield i_j_x

def populate_emb(e, words, dim, stdev=1):
    e_prime[words] = np.random.randn(len(words), args.dim) * stdev / np.sqrt(dim)
    e_prime[[f"{w}_c" for w in words]] = np.random.randn(len(words), dim) * stdev / np.sqrt(dim)
    return e_prime

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--std", type=float, default=1.0, help="Normalized standard deviation of the embeddings")
    parser.add_argument("--words", type=str, nargs="+", default=["soccer", "cookies", "about", "food", "democrats", "republicans"])
    parser.add_argument("--modelpath", type=str, default=None)
    parser.add_argument("--dim", type=int, default=100)
    args = parser.parse_args()
    print(args)

    words = args.words    
    e_prime = Embedding(set(words), dimensionality=args.dim)
    datapath = f'data/sgns_simulated-dim-{args.dim}-N-{args.N}.json'
    if args.modelpath is not None:
        print(f"Load model from {args.modelpath}...")
        e = Embedding(saved_model_path=args.modelpath)
        assert args.dim == e.theta.shape[1]
        e_prime = transfer_embeddings(e, e_prime)
    else:
        embpath = datapath.replace("data/", "trained/").replace(".json", ".pkl")
        e_prime = populate_emb(e_prime, words, args.dim, stdev=args.std)
        e_prime.save(embpath)

    batch_size = 500
    batches = args.N // batch_size
    d = {"i": [], "j": [], "x": []}
    for ix, data in enumerate(generate_dataset(e_prime, batches, batch_size)):
        i, j, x = data

        d["i"] = d["i"] + i
        d["j"] = d["j"] + j
        d["x"] = d["x"] + [int(x_i) for x_i in x.numpy()]

    with open(datapath, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


