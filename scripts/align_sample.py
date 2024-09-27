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


def main(args):


    df = pd.read_csv(args.path)
    print(df)
    e_ref = Embedding(saved_model_path=args.e_ref)
    print(e_ref)
    words = [wd for wd in e_ref.vocabulary if "_c" not in wd]

    # Create new dataframe
    rows = []
    columns = sorted(list(e_ref.vocabulary))

    for ix, row in progressbar.progressbar(df.iterrows()):
        e_ix = Embedding(set(words), dimensionality=e_ref.dimensionality)

        for col in df.columns:
            d = int(col.split("_")[-1])
            wd = "_".join(col.split("_")[:-1])
            x_wd = e_ix[wd].numpy()
            x_wd[d] = float(row[col])
            e_ix[wd] = x_wd


        e_ix = align(e_ref, e_ix, list(e_ref.vocabulary))
        vectors = e_ix[columns].numpy()
        newcols, row = [], []
        for dim in range(vectors.shape[-1]):
            newcols += [f"{wd}_{dim}" for wd in columns]
            row += [vectors[ix][dim] for ix, _ in enumerate(columns)]
        rows.append(row)

    new_df = pd.DataFrame(rows, columns=newcols)
    new_df.to_csv(args.path.replace(".csv", "-ortho.csv"), index=False)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--e_ref", type=str, required=True, help="Reference embedding")
    args = argparser.parse_args()
    print(args)
    main(args)
