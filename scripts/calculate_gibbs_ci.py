import polars as pl
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

WARMUP = 150
TRUEVAL = 0.654114695301912
dfs = []
for p in Path(".").glob("gibbs-samples*.csv"):

    print(p)
    df = pl.read_csv(p)
    if len(df) >= WARMUP:
        N = int(p.stem.split("-")[-1])
        #print(df)

        df = df.with_columns(pl.lit(N).alias("N"))
        df = df.with_row_index()
        dfs.append(df)

df = pl.concat(dfs)

WD1 = "word0"
WD2 = "word1"

df = df.with_columns((pl.col(f"{WD1}_0") * pl.col(f"{WD2}_c_0")
    + pl.col(f"{WD1}_1") * pl.col(f"{WD2}_c_1") ).alias("eta"))

df = df.with_columns((1.0/( 1.0 + pl.col("eta").neg().exp() )).alias("p"))
print(df)

df = df.filter(pl.col("index") >= WARMUP)
print(df)

sns.lineplot(df, x="N", y="p", errorbar=("pi", 90))
plt.axhline(y=TRUEVAL, color="orange")
plt.show()