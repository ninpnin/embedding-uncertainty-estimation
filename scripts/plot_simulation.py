import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def main(args):

    dfs = []
    for dfpath in args.dfs:
        N = int(dfpath.split("-N-")[-1].split("-")[0].split(".")[0])
        print(dfpath, N)
        df = pd.read_csv(dfpath)
        df["n"] = N
        #print(df)
        dfs.append(df)

    df = pd.concat(dfs)
    print(df)

    df = df[df["w1"] + "_c" != (df["w2"])]
    #df = df[( (df["w1"] == "dog")) | ( (df["w1"] == "cat")) ]
    #df = df[( (df["w1"] == "dog") & (df["w2"] == "saw_c")) | ( (df["w1"] == "the") & (df["w2"] == "cat_c")) ]
    df["combination"] = df["w1"] + " " + df["w2"]
    sns.lineplot(
        data=df, x="n", y="dot-estimate",
        hue="combination",# style="event",
        errorbar=("sd", 1)
        #kind="line"
    )
    plt.xscale('log')
    sns.set_style("white")
    plt.show()

    df = df[( (df["w1"] == "dog") & (df["w2"] == "saw_c")) ]#| ( (df["w1"] == "the") & (df["w2"] == "cat_c")) ]

    means = df[["w1", "w2", "n", "dot-estimate", "dot-true"]].groupby(["w1", "w2", "n"]).mean()
    means = means.reset_index()
    print(means)

    stds = df[["w1", "w2", "n", "dot-estimate", "dot-true"]].groupby(["w1", "w2", "n"]).std()
    stds = stds.reset_index()
    print(stds)


    estimation_stds = df[["w1", "w2", "n", "dot-estimate", "dot-true", "sampling"]].groupby(["w1", "w2", "n", "sampling"]).std()
    estimation_stds = estimation_stds.reset_index()
    estimation_stds = estimation_stds[["w1", "w2", "n", "dot-estimate", "dot-true"]].groupby(["w1", "w2", "n"]).mean()
    estimation_stds = estimation_stds.reset_index()

    estimation_stds["type"] = "estimation"
    stds["type"] = "total"
    print(estimation_stds)

    stds = pd.concat([stds, estimation_stds])

    # Plot the responses for different events and regions
    sns.lineplot(x="n", y="dot-estimate",
             hue="type",
             data=stds)
    plt.ylim(0, None)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfs", type=str, nargs="+")
    args = parser.parse_args()
    print(args)

    main(args)

