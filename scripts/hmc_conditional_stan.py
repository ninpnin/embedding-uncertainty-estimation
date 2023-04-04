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
import json

def conditional_hmc(wds, e, text, ws=5, ns_prime=1000):
    # positive samples
    i = []
    j = []
    freq = {}
    datalen = len(text)

    for ix, wd in enumerate(text):
        if wd in wds:
            context = text[ix-ws:ix] + text[ix+1:ix+ws+1]
            skip = False
            for wd_c in context:
                if wd_c not in e:
                    print("Skipping", context)
                    skip = True
            if skip:
                continue

            i.append(wd)
            j.append(context)

        freq[wd] = freq.get(wd, 0.0) + 1 / datalen

    j_neg = []
    
    freq_prime = {wd: f ** (3/4) for wd, f in freq.items()}
    freq_prime_sum = sum(freq_prime.values())
    freq_prime = {wd: f / freq_prime_sum for wd, f in freq_prime.items()}
    print(freq[wds[0]])
    print(freq_prime[wds[0]])
    # negative samples
    for _ in range(ns_prime):
        ix = np.random.randint(ws, len(text)-ws)
        context = text[ix-ws:ix] + text[ix+1:ix+ws+1]
        skip = False
        for wd_c in context:
            if wd_c not in e:
                print("Skipping", context)
                skip = True
        if skip:
            continue
        j_neg.append(context)


    j = tf.constant(j) + "_c"
    j_neg = tf.constant(j_neg)  + "_c"
    print(j.shape)
    print(j_neg.shape)

    print(j)
    e_j = e[j]
    e_j_neg = e[j_neg]

    print(e_j.shape)
    print(e_j_neg.shape)


    e_j = tf.reduce_sum(e_j, axis=1)
    e_j_neg = tf.reduce_sum(e_j_neg, axis=1)

    print(e_j.shape)
    print(e_j_neg.shape)

    i = np.array(i)
    return i, e_j, e_j_neg, freq, freq_prime

def save(i, e_j, e_j_neg, freq, freq_prime, args):
    print(i)
    print(e_j)

    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir()

    np.save(outdir / "i.npy", i)
    np.save(outdir / "e_j.npy", e_j)
    np.save(outdir / "e_j_neg.npy", e_j_neg)

    with (outdir / "freq.json").open("w") as f:
        freq = {wd: freq[wd] for wd in args.wds}
        json_str = json.dumps(freq, indent=4)
        f.write(json_str)

    with (outdir / "freq_prime.json").open("w") as f:
        freq_prime = {wd: freq_prime[wd] for wd in args.wds}
        json_str = json.dumps(freq_prime, indent=4)
        f.write(json_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns_prime", type=int, default=1000)
    parser.add_argument("--data", type=str, default="data/wikismall.txt")
    parser.add_argument("--outdir", type=str, default="data/stan/")
    parser.add_argument("--wds", type=str, nargs="+", default=["dog"])
    args = parser.parse_args()
    
    text = open(args.data).read().lower().split()
    text, vocabulary = preprocess_standard(text)
    print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

    e = Embedding(saved_model_path="./lapl_emb.pkl")
    #e = Embedding(vocabulary=vocabulary, dimensionality=100)

    print(e)
    # Perform MAP estimation
    wds = args.wds

    i, e_j, e_j_neg, freq, freq_prime = conditional_hmc(wds, e, text, ns_prime=args.ns_prime)
    save(i, e_j, e_j_neg, freq, freq_prime, args)




