import numpy as np
import cmdstanpy
import pickle, json
from trainerlog import get_logger
LOGGER = get_logger("conversion")
from probabilistic_word_embeddings.embeddings import Embedding
from pathlib import Path

def load_pickle_emb(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    LOGGER.debug(f"{type(d)}")
    LOGGER.debug(f"{d.__dir__()}")
    return d

def main(args):
    fit = load_pickle_emb(args.path)
    #vocab_size = fit["word_vectors"].shape[0]
    #dimensionality = fit["word_vectors"].shape[1]
    vocab_size = fit["word_vectors"].shape[0]
    dimensionality = fit["word_vectors"].shape[1]
    K, V = dimensionality, vocab_size

    samples = fit["word_vectors"].shape[2]

    LOGGER.info(f"V: {vocab_size}, K: {dimensionality}, samples: {samples}")
    #vocab = sorted([f"word{ix}" for ix in range(vocab_size)])
    vocab = [f"word{ix}" for ix in range(vocab_size)]
    print(" ".join(vocab))
    vocab_c = [f"{wd}_c" for wd in vocab]
    LOGGER.debug(f"Vocabulary: {vocab}")

    LOGGER.info(f"Convert to pwe.Embedding...")
    outfolder = args.outpath
    if args.outpath is None:
        random_nums = "".join([str(digit).split(".")[-1][0] for digit in fit["word_vectors"].flatten().tolist()[:6]])
        outfolder = f"hmc-stan-K{K}-V{V}-embedding-{random_nums}"

    outfolder = Path(outfolder)
    LOGGER.info(f"Output folder {outfolder}...")

    for sample_ix in range(samples):
        e = Embedding(set(vocab), dimensionality=dimensionality)
        for ix, pair in enumerate(zip(vocab, vocab_c)):
            wd, wd_c = pair
            #LOGGER.debug(f"Copy {wd} and {wd_c} (index {ix})...")
            e[wd] = fit["word_vectors"][ix, :, sample_ix]
            e[wd_c] = fit["context_vectors"][ix, :, sample_ix]
        
        LOGGER.info(f"K: {e.dimensionality}")
        outpath = outfolder / f"sample-{sample_ix}.pkl"
        outfolder.mkdir(exist_ok=True)
        LOGGER.info(f"Save to {outpath}...")
        e.save(str(outpath.absolute()))

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--outpath", type=str, default=None)
    args = argparser.parse_args()
    print(args)
    main(args)
