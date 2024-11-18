import hashlib
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
from probabilistic_word_embeddings.embeddings import Embedding
import tqdm
import tensorflow as tf

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def zipf_freqs(vocab, offset=0.0, temperature=1.0):
    words = sorted([wd for wd in vocab if "_c" not in wd])
    plain_freqs = [(1.0 + ix + offset) ** (- temperature) for ix in range(len(words))]
    plain_freqs = np.array(plain_freqs)
    freqs = plain_freqs / np.sum(plain_freqs)
    freqs_dict = {wd: freqs[ix] for ix, wd in enumerate(words)}
    print(freqs_dict)
    return freqs_dict

def get_random_word(freqs, size=1):
    words = [wd for wd in freqs]
    freqlist = [freqs[wd] for wd in words]
    return list(np.random.choice(words, p=freqlist, size=size))

def main(args):
    
    with open(args.ref_data) as f:
        d = json.load(f)

    print(d.keys())
    vocab = d["vocabulary"]

    freqs = zipf_freqs(vocab, offset=args.b, temperature=args.a)
    theta = np.array(d["theta"])
    V, K = theta.shape
    V = V // 2 # there are 2K vectors in the full embedding (theta) = (alpha, rho)

    e = Embedding(set(vocab), dimensionality=K)

    for wd in vocab:
        e[f"{wd}"] = theta[vocab[wd]]
        e[f"{wd}_c"] = theta[vocab[wd] + V]

    #print(e.theta.numpy())
    #print(np.array_str(e.theta.numpy(), precision=2, suppress_small=True))

    seed = None
    if args.seed is None:
        seed = np.random.randint(2 ** 32 - 1)
    else:
        seed = md5hash(args.seed)
    
    print("seed:", seed)
    np.random.seed(seed)

    w = [str(wd) for wd in get_random_word(freqs, size=args.N)]
    v = [str(wd) for wd in get_random_word(freqs, size=args.N)]
    x = []

    chunk_size = 1000
    rng = tf.random.Generator.from_seed(seed=seed)

    for i in tqdm.tqdm(range(args.N // chunk_size)):
        i = i * chunk_size
        w_i = w[i : i+chunk_size]
        v_i = v[i : i+chunk_size]

        e_w_i = e[w_i]
        e_v_i = e[v_i]

        etas = tf.reduce_sum(tf.multiply(e_w_i, e_v_i), axis=1)
        
        p_i = sigmoid(etas)#sigmoid(np.dot(e_w_i, e_v_i))
        p_i = np.array(p_i)

        # + 0.0 converts to float
        x_i = list((np.random.rand(chunk_size) < p_i) + 0.0)
        
        x += x_i





if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--ref_data", type=str, required=True)
    argparser.add_argument("--outpath", type=str, default=None)
    argparser.add_argument("--N", type=int, default=2000000)
    argparser.add_argument("--a", type=float, default=1.0, help="Zipf law exponent parameter, usually roughly 1.0")
    argparser.add_argument("--b", type=float, default=2.7, help="Zipf law offset parameter, 2.7 is a common value for English")
    argparser.add_argument("--seed", type=int, default=None)
    args = argparser.parse_args()
    main(args)