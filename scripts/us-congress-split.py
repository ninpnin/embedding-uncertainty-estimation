import polars as pl
import numpy as np
from pathlib import Path
import json
import progressbar
import random

def main(args):
    with open(args.datapath, "r") as f:
        d = json.load(f)
    print(d.keys())
    data = d["data"]
    random.Random(args.seed).shuffle(data)
    print(data[:10])
    
    start, end = 0, None
    N = len(data)
    for split, size in zip(["train", "val", "test"], args.split):
        if end is not None:
            start = end
        end = start + int(size * N)
        print("split,", split, "start", start, "end", end)
        data_split = data[start:end]
        d_split = {}
        d_split["vocabulary"] = d["vocabulary"]
        d_split["data"] = data_split
        
        filename = args.datapath.replace(".json", f"-{split}.json")
        print(filename)
        with open(filename, "w") as f:
            json.dump(d_split, f, indent=0, ensure_ascii=False)


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--datapath", type=str, required=True)
    argparser.add_argument("--split", type=float, nargs="+", default=[0.7, 0.15, 0.15])
    argparser.add_argument("--seed", type=int, default=None)
    args = argparser.parse_args()
    print(args)
    main(args)
