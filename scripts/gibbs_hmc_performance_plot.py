from trainerlog import get_logger
LOGGER = get_logger("perf-plot")
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
plt.rc('font',  size=14) #family='serif',

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--logplot", type=bool, default=False)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    model = "sgns"
    if "cbow" in args.data_path:
        model = "cbow"

    df = pl.read_csv(args.data_path)
    print(df)
    df = df.with_columns(pl.col("N") * 1000)
    df = df.with_columns((pl.col("runtime") / 60).alias("runtime (minutes)"))

    #plt.figure(figsize=(10, 6))
    try:
        g = sns.lineplot(df, x="N", y="runtime (minutes)", hue="method",linewidth=2.5,  palette=["tab:blue", "tab:green"])
    except:
        import pandas as pd
        df = df.to_pandas()
        g = sns.lineplot(df, x="N", y="runtime (minutes)", hue="method", linewidth=2.5, palette=["tab:blue", "tab:green"])

    
    if args.logplot:
        plt.xscale('log')
        plt.yscale('log')
    sns.despine()
    plt.ylabel(None, fontsize=24)

    TICKSIZE = 15
    plt.xticks(fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)

    import matplotlib.ticker as ticker
    g.xaxis.set_major_formatter(ticker.EngFormatter())
    #g.yaxis.set_major_formatter(ticker.EngFormatter())
    
    g.set_ylabel('runtime (minutes)', labelpad=15, fontsize=15)
    plt.tight_layout()
    plt.grid()
    image_filepath = f"img/speed-benchmark-gibbs-hmc-{model}.pdf"
    if args.logplot:
        image_filepath = image_filepath.replace(".pdf", "-log.pdf")
    plt.savefig(image_filepath)
    plt.show()