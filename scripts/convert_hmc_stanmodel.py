"""
Convert pickled stan fit file to a folder of pwe.Embedding files
"""
import numpy as np
import cmdstanpy
import pickle, json
from trainerlog import get_logger
LOGGER = get_logger("conversion")
from probabilistic_word_embeddings.embeddings import Embedding
from pathlib import Path
import tqdm

def load_pickle_emb(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    LOGGER.debug(f"{type(d)}")
    LOGGER.debug(f"{d.__dir__()}")
    return d

def main(args):
    fit = load_pickle_emb(args.path)
    #vocab_size = fit.word_vectors.shape[0]
    #dimensionality = fit.word_vectors.shape[1]

    word_vectors = fit.word_vectors
    context_vectors = fit.context_vectors

    if args.transpose_vecs:
        word_vectors = word_vectors.transpose(1,2,0)
        context_vectors = context_vectors.transpose(1,2,0)

    vocab_size = word_vectors.shape[0]
    dimensionality = word_vectors.shape[1]
    K, V = dimensionality, vocab_size

    samples = word_vectors.shape[2]

    if samples < 200:
        LOGGER.error(f"SUSPICIOUSLY LOW sample count {samples}")

    LOGGER.info(f"V: {vocab_size}, K: {dimensionality}, samples: {samples}")
    #vocab = sorted([f"word{ix}" for ix in range(vocab_size)])
    vocab = None
    if args.vocabulary is None:
        LOGGER.info(f"Create simulation vocabulary: (word0, word1, ..., wordV)")
        LOGGER.warning(f"The previous version sorted by int order 2 < 19, this one by string order 2 > 19")
        LOGGER.warning(f"Double check this when interpretting and plotting results")
        vocab = sorted([f"word{ix}" for ix in range(vocab_size)])
        LOGGER.debug(" ".join(vocab))
    else:
        LOGGER.info(f"Load vocabulary from file {args.vocabulary}")
        with open(args.vocabulary, "r") as f:
            datafile = json.load(f)

        vocab_dict = bidict.bidict(datafile["vocabulary"])
        vocab = sorted([(a,b) for a,b in vocab_dict.items()], key=lambda x: x[1])
        vocab = [a for a,b in vocab]


    LOGGER.debug("Vocab: " + " ".join(vocab))
    vocab_c = [f"{wd}_c" for wd in vocab]
    LOGGER.debug(f"Vocabulary: {vocab}")

    LOGGER.info(f"Convert to pwe.Embedding...")
    outfolder = args.outpath
    if args.outpath is None:
        random_nums = "".join([str(digit).split(".")[-1][0] for digit in word_vectors.flatten().tolist()[:6]])
        if "mala" in args.path:
            outfolder = f"mala-stan-K{K}-V{V}-embedding-{random_nums}"
        else:
            outfolder = f"hmc-stan-K{K}-V{V}-embedding-{random_nums}"

    outfolder = Path(outfolder)
    LOGGER.info(f"Output folder {outfolder}...")

    for sample_ix in tqdm.tqdm(range(samples)):
        e = Embedding(set(vocab), dimensionality=dimensionality)
        for ix, pair in enumerate(zip(vocab, vocab_c)):
            wd, wd_c = pair
            #LOGGER.debug(f"Copy {wd} and {wd_c} (index {ix})...")
            e[wd] = word_vectors[ix, :, sample_ix]
            e[wd_c] = context_vectors[ix, :, sample_ix]
        
        LOGGER.debug(f"K: {e.dimensionality}")
        outpath = outfolder / f"sample-{sample_ix}.pkl"
        outfolder.mkdir(exist_ok=True)
        LOGGER.debug(f"Save to {outpath}...")
        e.save(str(outpath.absolute()))

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True, help="Path to the stan fit file")
    argparser.add_argument("--vocabulary", type=str, default=None)
    argparser.add_argument("--outpath", type=str, default=None, help="Path to the output folder; optional")
    argparser.add_argument("--transpose_vecs", type=bool, default=False, help="Transpose word_vectors axes so that you get (V,K, samples)")
    args = argparser.parse_args()
    print(args)
    main(args)
