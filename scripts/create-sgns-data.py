from pathlib import Path
import polars as pl
import progressbar
import json
import numpy as np

folder = Path("raw-speeches")

VOCAB_SIZE = 5000
WS = 2
NS = 1

with open("wc-stemmed.json") as f:
    wordcounts = json.load(f)
 
wordcounts = pl.DataFrame(wordcounts, orient="col").transpose(include_header=True, header_name="word", column_names=["count"])
wordcounts = wordcounts.sort("count", descending=True)
print(wordcounts)

wordcounts = wordcounts.filter(pl.col("word").str.contains("\{").is_not())
wordcounts = wordcounts.filter(pl.col("word").str.contains("\}").is_not())
wordcounts = wordcounts.head(VOCAB_SIZE)
wordcounts = wordcounts.with_columns((pl.col("count") / pl.col("count").sum()).alias("frequency"))
print(wordcounts)

vocab = set(wordcounts["word"])

vocab_dict = {k: val for val, k in enumerate(sorted(wordcounts["word"]))}

def get_ns(n):
    return [str(wd) for wd in np.random.choice(list(wordcounts["word"]), size=n, p=list(wordcounts["frequency"]))]



for file_ix, p in enumerate(sorted(folder.glob("stemmed-*.ndjson"))):
    df_p = pl.read_ndjson(p)
    
    print(df_p)
    
    data = []
    for t in progressbar.progressbar(df_p["speech"]):
        text = t.split()
        N = len(text)
        
        #exit()
        negative_samples = get_ns(NS * WS * N * 2)
        ns_ix = 0
        for i in range(0, N):
            w = text[i]
            if w in vocab:
                for j in range(max(0, i - WS), min(N, i + WS + 1)):
                    if i != j and text[j] in vocab:
                        v = text[j]
                        data.append({"w": w, "v": v, "x": 1})
                    
                        for _ in range(NS):
                            data.append({"w": w, "v": negative_samples[ns_ix], "x": 0})
                            ns_ix += 1
        
        #print(file_ix, p.stem)
    
    data_dict = {"raw-file": p.stem, "vocabulary": vocab_dict, "data": data}
    filename = f"congress-stemmed-ws-{WS}-ns-{NS}-vocab-{VOCAB_SIZE}-{file_ix}.json"
    with open(f"processed/{filename}", "w") as f:
        json.dump(data_dict, f, indent=0)
        #with open(f"{p.stem}")
        #print(data)


        
#exit()
