import polars as pl
import sys
from pathlib import Path
import socket
hostname = socket.gethostname()

laplace = "laplace" in " ".join(sys.argv)
df = None
if not laplace:
    print("Gibbs (probably)")
    df = pl.read_csv("logs/coverage.csv")
else:
    print("Laplace (probably)")
    df = pl.read_csv("logs/laplace-results.csv")
    df = df.with_columns(pl.col("coverage").alias("ci-90-coverage"))
    
rmse_results = df.group_by("V", "K", "N").agg([
    pl.col('RMSE').mean().alias('RMSE'),
    pl.col('RMSE_normalized').mean().alias('RMSE_normalized'),
    pl.count('RMSE').alias('no_of_datasets')
])
rmse_results = rmse_results.sort("K", "V", "N")
print(rmse_results)

if laplace:
    rmse_results = rmse_results.with_columns(pl.lit("Laplace").alias("method"))
else:
    rmse_results = rmse_results.with_columns(pl.lit("Gibbs").alias("method"))

rmse_filename = f"logs/rmse-{hostname}.csv"
rmse_path = Path(rmse_filename)
if rmse_path.exists():
    rmse_results_old = pl.read_csv(rmse_filename)
    rmse_results = pl.concat([rmse_results, rmse_results_old], how="vertical_relaxed")
    rmse_results = rmse_results.unique()
    rmse_results = rmse_results.sort("method", "K", "V", "N")
    
rmse_results.write_csv(rmse_filename)
    

for K in [5, 10, 20]:
    for V in [100, 200]:
        df_KV = df.filter(pl.col("K") == K).filter(pl.col("V") == V)
        df_KV = df_KV.with_columns(pl.col("ci-90-coverage") * 100.0)
        
        #print(df_KV)
        print(f"K: {K}, V: {V}")
        if len(df_KV) > 0:
            df_t = df_KV.group_by("N").mean()
            print(f"{len(df_KV)} datasets")
            df_t = df_t.with_columns(pl.format("{}", pl.col("ci-90-coverage").round(1)).alias("ci-90-coverage"))
            df_t = df_t.sort("N").transpose(include_header=True)
            #print(df_t)
            row = df_t.filter(pl.col("column") == "N").row(0)
            row = " & ".join([str(val) for val in row])
            print(row)

            row = df_t.filter(pl.col("column") == "ci-90-coverage").row(0)
            row = " & ".join([val for val in row])
            print(row)
        print()
            

