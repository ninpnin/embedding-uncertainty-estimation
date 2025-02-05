import numpy as np
import cmdstanpy
import pickle, json
from trainerlog import get_logger
LOGGER = get_logger("conversion")

def load_pickle_emb(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    LOGGER.debug(f"{type(d)}")
    LOGGER.debug(f"{d.__dir__()}")
    return d

def main(args):
    fit = load_pickle_emb(args.path)
    vocab_size = fit.word_vectors.shape[0]
    dimensionality = fit.word_vectors.shape[1]

    LOGGER.info(f"V: {vocab_size}, K: {dimensionality}")
    vocab = sorted([f"word{ix}" for ix in range(vocab_size)])
    print(" ".join(vocab))
    vocab_c = [f"{wd}_c" for wd in vocab]
    LOGGER.debug(f"Vocabulary: {vocab}")

    if args.format == "json":
        LOGGER.info(f"Convert to JSON...")
        theta = np.zeros((vocab_size*2, dimensionality))

        for ix, pair in enumerate(zip(vocab, vocab_c)):
            wd, wd_c = pair
            theta[ix] = fit.word_vectors[ix]
            theta[ix+vocab_size] = fit.context_vectors[ix]

        #print(theta)

        full_vocab = vocab + vocab_c
        full_vocab = {wd: ix for ix, wd in enumerate(full_vocab)}
        #print(full_vocab)

        theta = theta.tolist()

        d = {"theta": theta, "vocabulary": full_vocab, "lambda0": 1.0}


        if args.outpath is None:
            args.outpath = args.path.replace(".pkl", ".json")

        LOGGER.info(f"Save to {args.outpath}...")
        with open(args.outpath, "w") as f:
            json.dump(d,f, indent=1, ensure_ascii=False)
    elif args.format == "embedding":
        LOGGER.info(f"Convert to pwe.Embedding...")
        from probabilistic_word_embeddings.embeddings import Embedding
        e = Embedding(set(vocab), dimensionality=dimensionality)
        for ix, pair in enumerate(zip(vocab, vocab_c)):
            wd, wd_c = pair
            LOGGER.debug(f"Copy {wd} and {wd_c} (index {ix})...")
            e[wd] = fit.word_vectors[ix]
            e[wd_c] = fit.context_vectors[ix]
        
        LOGGER.info(f"Save to {args.outpath}...")
        LOGGER.info(f"K: {e.dimensionality}")
        e.save(args.outpath)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--outpath", type=str, default=None)
    argparser.add_argument("--format", type=str, default="json", choices=['json', 'embedding'])
    args = argparser.parse_args()
    print(args)
    main(args)
