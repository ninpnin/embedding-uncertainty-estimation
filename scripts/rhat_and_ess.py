import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import random
import progressbar
import pandas as pd
import arviz as az

def calculate_rhat(dataframe, elem):
    x = np.array(dataframe[elem])
    L = len(x) // 2
    x_start = x[:L]
    x_end = x[-L:]

    #print(x_start, x_end)
    #print(x_start.shape, x_end.shape)
    x_prime = np.array([x_start, x_end])
    if np.std(x) <= 0.00001:
        return 1.0

    return az.rhat(x_prime)

def calculate_ess(dataframe, elem):
    x = np.array(dataframe[elem])
    if np.var(x) == 0.0:
        return len(x)
    return az.ess(x)

def calculate_mean_std_mcse(dataframe, elem):
    ess = calculate_ess(dataframe, elem)
    x = np.array(dataframe[elem])

    e_x = np.mean(x)
    e_x2 = np.mean(x * x)

    var = e_x2 - (e_x ** 2)
    return e_x, np.sqrt(var), np.sqrt(var / ess)

def get_df_stats(samples):
    cols = samples.columns
    data = {"elem": list(cols), "rhat": [], "ess": [], "mean": [], "std": [], "mcse": []}
    for col in progressbar.progressbar(cols):
        rhat_col = calculate_rhat(samples, col)
        data["rhat"] = data["rhat"] + [rhat_col]

        ess_col = calculate_ess(samples, col)
        data["ess"] = data["ess"] + [ess_col]

        mean_col, std_col, mcse_col = calculate_mean_std_mcse(samples, col)
        data["mean"] = data["mean"] + [mean_col]
        data["std"] = data["std"] + [std_col]
        data["mcse"] = data["mcse"] + [mcse_col]

    df = pd.DataFrame(data)
    df["ESS/N"] = df["ess"] / len(samples)
    df["converged-1.01"] = df["rhat"] <= 1.01
    df["converged-1.05"] = df["rhat"] <= 1.05
    print(df)

    print("Mean ESS", df["ess"].mean())
    print("Median ESS", df["ess"].median())
    print("Converged %", df["converged-1.01"].mean(), "Rhat <= 1.01")
    print("Converged %", df["converged-1.05"].mean(), "Rhat <= 1.05")

    return df["ess"].median()

def get_emb_df(path, burnin):
    from probabilistic_word_embeddings.embeddings import Embedding
    K = None
    sample_folder = Path(path)
    samples = sorted(sample_folder.glob("*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
    # By default the first half
    if burnin is None:
        samples = samples[len(samples) // 2:]
    else:
        samples = samples[burnin:]

    rows = []
    vocab = None
    for sample in progressbar.progressbar(samples):
        e = Embedding(saved_model_path=str(sample.absolute()))
        K = e.dimensionality
        vocab = sorted([wd for wd in e.vocabulary])
        theta = e[vocab].numpy().flatten().tolist()
        rows.append(theta)

    columns = []
    for wd in vocab:
        for k in range(K):
            columns.append(f"{wd}_{k}")
    df = pd.DataFrame(rows, columns=columns)
    print(df)
    return df

def main(args):
    ess_medians = []
    for path in args.path:
        if ".csv" in path:
            samples = pd.read_csv(path)
            if args.burnin is None:
                samples = samples.tail(len(samples) // 2)
            else:
                samples = samples.tail(len(samples) - args.burnin)
            ess_median = get_df_stats(samples)
        else:
            samples = get_emb_df(path, args.burnin)
            ess_median = get_df_stats(samples)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, nargs="+", required=True, help="Path to the folder with sample-IX.pkl embedding files")
    argparser.add_argument("--burnin", type=int, default=None, help="Number of iterations to be discarded")
    args = argparser.parse_args()
    print(args)
    main(args)
