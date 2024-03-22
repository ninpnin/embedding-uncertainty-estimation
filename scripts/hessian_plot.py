import polars as pl
import progressbar
import json
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
def main(args):
    df = pl.read_csv(args.samplepath)
    WS = args.ws
    print(df)
    words = set(df["word"])
    print(words)
    """
    with open(args.datapath) as f:
        s = f.read().lower().split()

    co_occurences = {}
    N = len(s)
    for i in progressbar.progressbar(range(N)):
        w = s[i]
        for j in range(i-WS, i+1+WS):
            if j != i and j >= 0 and j < N:
                v = s[j]

                if w in words and v in words:
                    pair1 = f"{w} {v}"
                    pair2 = f"{v} {w}"
                    co_occurences[pair1] = co_occurences.get(pair1, 0) + 1
                    co_occurences[pair2] = co_occurences.get(pair2, 0) + 1
                
    print(co_occurences)
    outpath = args.samplepath.replace("/", "-").replace(".", "")
    with open(f"output-{outpath}.json", "w") as file:
    # Write the data to the file in JSON format
        json.dump(co_occurences, file, indent=4)  # Optional: indent for readability
    """
    outpath = args.samplepath.replace("/", "-").replace(".", "")
    with open(f"output-{outpath}.json") as file:
    # Write the data to the file in JSON format
        co_occurences = json.load(file)  # Optional: indent for readability
    print(co_occurences)

    rows = []
    for wordpair, count in co_occurences.items():
        w,v = wordpair.split()
        row = [w, v, 1]
        rows.append(row)

    df = pd.DataFrame(rows, columns=["w", "v", "count"])
    print(df)
    pivoted = df.pivot(index="w", columns="v", values="count").fillna(0.0)
    print(pivoted)

    sns.heatmap(pivoted, center=0.0, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.show()


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--datapath", type=str)
    argparser.add_argument("--samplepath", type=str)
    argparser.add_argument("--ws", type=int, default=5)
    args = argparser.parse_args()
    print(args)
    main(args)