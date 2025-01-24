import numpy as np
import cmdstanpy
import pickle, json
from pathlib import Path

def load_pickle_emb(path):
    with open(path, "rb") as f:
        fit = pickle.load(f)

    return fit

def main(args):
    fit = load_pickle_emb(args.path)
    vocab_size, dimensionality = fit.dims[0]
    print("V", vocab_size, "K",dimensionality)
    #dimensionality = fit.word_vectors.shape[1]
    df_fit = fit.to_frame()

    df_rho_cols = [col for col in df_fit.columns if "word_vector" in col]
    df_alpha_cols = [col for col in df_fit.columns if "context_vector" in col]

    mapping_dict = {}
    newcols = []
    for w in range(vocab_size):
        for k in range(dimensionality):
            # word_vectors.98.5

            word_old = f"word_vectors.{w+1}.{k+1}"
            context_old = f"context_vectors.{w+1}.{k+1}"

            # word0_c_0,word1_0,word10_0,word10_c_0
            word_new = f"word{w}_{k}"
            context_new = f"word{w}_c_{k}"
            newcols.append(word_new)
            newcols.append(context_new)

            mapping_dict[word_old] = word_new
            mapping_dict[context_old] = context_new

    df_fit = df_fit.rename(columns=mapping_dict)
    newcols = sorted(newcols) 
    print(df_fit)

    df_fit = df_fit[newcols]
    print(df_fit)

    filepath = Path(args.path)
    filename = filepath.stem
    print(filepath.parent)
    print(filename)

    outpath = filepath.parent / f"{filename}.csv"
    df_fit.to_csv(outpath)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--outpath", type=str, default=None)
    args = argparser.parse_args()
    print(args)
    main(args)
