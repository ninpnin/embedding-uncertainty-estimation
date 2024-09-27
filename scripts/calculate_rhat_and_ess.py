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

def main(args):
    df = pd.read_csv(args.path)
    samples = df.tail(len(df) - args.burnin)
    print(samples)

    cols = samples.columns
    pass

    data = {"elem": list(cols), "rhat": [], "ess": []}
    for col in cols:
        rhat_col = calculate_rhat(samples, col)
        data["rhat"] = data["rhat"] + [rhat_col]

        ess_col = calculate_ess(samples, col)
        data["ess"] = data["ess"] + [ess_col]

    df = pd.DataFrame(data)
    print(df)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--burnin", type=int, default=500)
    args = argparser.parse_args()
    print(args)
    main(args)
