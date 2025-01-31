import numpy as np
import tqdm
from pathlib import Path
import random, json
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default=[], nargs="+")
    args = parser.parse_args()

    dfs = []
    for results_file in args.results_file:
        df = pl.read_csv(results_file)
        print(df)
        dfs.append(df)
    df = pl.concat(dfs)
    df = df.with_columns((pl.col("N") / pl.col("V")).alias("N_per_V"))
    df = df.with_columns((pl.col("N") / (pl.col("K") * pl.col("K") * pl.col("V"))).alias("N_per_KKV"))
    df = df.with_columns(pl.concat_str("K", "V", separator="-").alias("K-V"))

    sns.lineplot(df, x="N", y="RMSE_normalized", hue="K-V")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)

    plt.show()