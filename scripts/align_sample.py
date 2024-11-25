import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import random
import progressbar
import pandas as pd
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.utils import align
#align(e_reference, e, words, words_reference=None):

def extract_last_sample(df):
    vocab = set()
    dimensionality = 0
    for col in df.columns:
        d = int(col.split("_")[-1])
        wd = "_".join(col.split("_")[:-1])
        vocab.add(wd)
        dimensionality = max(d + 1, dimensionality)

    words = [wd for wd in list(vocab) if "_c" not in wd]
    e_ref = Embedding(set(words), dimensionality=dimensionality)
    columns = sorted(list(e_ref.vocabulary))
    row = df.iloc[len(df)-1]
    e_ref[columns] = np.array(row).reshape((e_ref.theta.shape))
    return e_ref

def main(args):


    df = pd.read_csv(args.path)
    print(df)

    e_ref = None
    if args.e_ref is None:
        e_ref = extract_last_sample(df)
    else:
        e_ref = Embedding(saved_model_path=args.e_ref)
    print(e_ref)
    words = [wd for wd in e_ref.vocabulary if "_c" not in wd]

    # Create new dataframe
    rows = []
    columns = sorted(list(e_ref.vocabulary))

    newcols = []
    for dim in range(e_ref.dimensionality):
        newcols += [f"{wd}_{dim}" for wd in columns]

    for ix, row in progressbar.progressbar(df.iterrows()):
        e_ix = Embedding(set(words), dimensionality=e_ref.dimensionality)
        e_ix[columns] = np.array(row).reshape((e_ix.theta.shape))
        e_ix = align(e_ref, e_ix, list(e_ref.vocabulary))

        vectors = e_ix[columns].numpy()
        newrow = list(vectors.flatten())
        rows.append(newrow)

    new_df = pd.DataFrame(rows, columns=newcols)
    new_df.to_csv(args.path.replace(".csv", "-ortho.csv"), index=False)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--e_ref", type=str, required=None, help="Reference embedding")
    args = argparser.parse_args()
    print(args)
    main(args)
