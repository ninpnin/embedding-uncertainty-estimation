import numpy as np
import cmdstanpy
import pickle, json

def load_pickle_emb(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    print(d.__dir__())
    return d

def main(args):
    fit = load_pickle_emb(args.path)
    print(fit.word_vectors)
    print(fit.context_vectors)
    vocab_size = fit.word_vectors.shape[0]
    dimensionality = fit.word_vectors.shape[1]

    print("vocab_size, dimensionality")
    print(vocab_size, dimensionality)
    vocab = [f"word{ix}" for ix in range(vocab_size)]
    vocab_c = [f"{wd}_c" for wd in vocab]
    print(vocab)

    theta = np.zeros((vocab_size*2, dimensionality))

    for ix, pair in enumerate(zip(vocab, vocab_c)):
        wd, wd_c = pair
        theta[ix] = fit.word_vectors[ix]
        theta[ix+vocab_size] = fit.context_vectors[ix]

    print(theta)

    full_vocab = vocab + vocab_c
    full_vocab = {wd: ix for ix, wd in enumerate(full_vocab)}
    print(full_vocab)

    theta = theta.tolist()

    d = {"theta": theta, "vocabulary": full_vocab, "lambda0": 1.0}


    if args.outpath is None:
        args.outpath = args.path.replace(".pkl", ".json")

    with open(args.outpath, "w") as f:
        json.dump(d,f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--outpath", type=str, default=None)
    args = argparser.parse_args()
    print(args)
    main(args)
