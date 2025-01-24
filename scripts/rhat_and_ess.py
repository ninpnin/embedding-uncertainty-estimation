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
    x_start = x[:len(x) // 2]
    x_end = x[-len(x) // 2:]

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

def main(args):

    ess_medians = []
    for path in args.path:
        df = pd.read_csv(path)
        samples = df.tail(len(df) - args.burnin)
        print(samples)

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

        ess_medians.append(df["ess"].median())


    print("Mean of median ESS", np.mean(ess_medians))
    print("Stdeb of median ESS", np.std(ess_medians))
    print("vals", "\n".join(ess_medians))


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, nargs="+", required=True)
    argparser.add_argument("--burnin", type=int, default=500)
    args = argparser.parse_args()
    print(args)
    main(args)
