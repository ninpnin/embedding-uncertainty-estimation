import polars as pl

df = pl.read_csv("logs/coverage.csv")
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
            row = ", ".join([str(val) for val in row])
            print(row)

            row = df_t.filter(pl.col("column") == "ci-90-coverage").row(0)
            row = ", ".join([val for val in row])
            print(row)
        print()
            

