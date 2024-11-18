import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json

def zipf_freqs(vocab, offset=0.0, temperature=1.0):
    words = sorted([wd for wd in vocab if "_c" not in wd])
    plain_freqs = [(1.0 + ix + offset) ** (- temperature) for ix in range(len(words))]
    plain_freqs = np.array(plain_freqs)
    freqs = plain_freqs / np.sum(plain_freqs)
    freqs_dict = {wd: freqs[ix] for ix, wd in enumerate(words)}
    print(freqs_dict)
    return freqs_dict

def main(args):
    
    with open(args.ref_data) as f:
        d = json.load(f)

    print(d.keys())
    vocab = d["vocabulary"]
    print(vocab)

    freqs = zipf_freqs(vocab, offset=args.b, temperature=args.a)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--ref_data", type=str, required=True)
    argparser.add_argument("--outpath", type=str, default=None)
    argparser.add_argument("--N", type=int, default=2000000)
    argparser.add_argument("--a", type=float, default=1.0, help="Zipf law exponent parameter, usually roughly 1.0")
    argparser.add_argument("--b", type=float, default=2.7, help="Zipf law offset parameter, 2.7 is a common value for English")
    args = argparser.parse_args()
    main(args)