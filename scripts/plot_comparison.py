import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def main(path):
    means_and_stds = pd.read_csv(path)
    print(means_and_stds)
    rows = []

    for _, row in means_and_stds.iterrows():
        print(row)

        wd = row["word"]
        mean = row["mean"]
        std = row["std"]
        method = row["method"]

        for i in range(1000):
            y = mean + np.random.randn() * std

            if wd != "dog":
                rows.append([wd, y, method])


    df = pd.DataFrame(rows, columns=["word", "similarity", "method"])

    sns.catplot(
        data=df, x="method", y="similarity", col="word",
        kind="violin"
    )    
    plt.show()

if __name__ == '__main__':
    path = "prelim_results.csv"
    main(path)