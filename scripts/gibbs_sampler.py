from embedding_uncertainty import embedding_gibbs
from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import copy
import json
import progressbar
import pandas as pd
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
import seaborn as sns
from matplotlib import pyplot as plt

def main(args):
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
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]

    vocab = set()
    if args.datapath is not None:
        with open(args.datapath, "rb") as f:
            d = json.load(f)

        LOGGER.debug(f"Keys: {d.keys()}")

        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            data.append((w,v,x))

    if args.data_len is not None:
        data = data[:args.data_len]

    x, y = [], []
    e = Embedding(vocab, dimensionality=2)
    
    rows = []
    columns = sorted(list(e.vocabulary))
    for sample_ix, e_sample in enumerate(embedding_gibbs(e, data, rounds=args.samples, yield_every=1)):
        print(e_sample)
        word0sample = e_sample["word0"].numpy()
        print(e_sample["word0"])
        x.append(word0sample[0])
        y.append(word0sample[1])

        vectors = e_sample[columns].numpy()
        print(vectors.shape)

        newcols, row = [], []
        for dim in range(vectors.shape[-1]):
            newcols += [f"{wd}_{dim}" for wd in columns]
            row += [vectors[ix][dim] for ix, _ in enumerate(columns)]
        rows.append(row)

        if sample_ix % 5 == 0:
            df = pd.DataFrame(rows, columns=newcols)
            df = df[sorted(newcols)]
            print(df)
            df.to_csv("gibbs-samples.csv", index=False)

    sns.set_theme()
    sns.lineplot(x=x, y=y, sort=False)
    plt.savefig(f"gibbsample-word0-{args.samples}.png")
    plt.show()