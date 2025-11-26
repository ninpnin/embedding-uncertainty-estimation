import numpy as np
#import stan
import pickle, json
from trainerlog import get_logger
LOGGER = get_logger("conversion")
import bidict
from probabilistic_word_embeddings.embeddings import Embedding
import tqdm
from pathlib import Path

def load_pickle_emb(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    LOGGER.debug(f"{d.__dir__()}")
    return d

def main(args):
    fit = load_pickle_emb(args.path)
    LOGGER.debug(f"{fit.column_names[:10]}")

    ij = [col.split("word_vectors[")[-1].split("]")[0] for col in fit.column_names if "word_vectors" in col]
    i = [int(elem.split(",")[0]) for elem in ij]
    j = [int(elem.split(",")[1]) for elem in ij]

    V = max(i)
    K = max(j)
    LOGGER.info(f"K: {K}, V: {V}")

    vocab = bidict.bidict({ix: f"word{ix}" for ix in range(V)})
    if args.vocab is not None:
        with open(args.vocab) as f:
            vocabdict = json.load(f)
        vocab = bidict.bidict(vocabdict["vocabulary"])
        print(list(vocab.items())[:10])
        #vocab = bidict.bidict({ix: wd for ix, wd in enumerate(args.vocab)})

    words = [vocab.inv[ix] for ix in range(V)]
    contexts = [vocab.inv[ix] + "_c" for ix in range(V)]

    theta_mean = fit._variational_mean[3:]
    #theta_mean = fit._variational_mean[3:]
    #print(fit._variational_sample.shape)

    sample = fit.variational_sample[:, 3:]
    print(sample.shape)
    N = sample.shape[0]
    LOGGER.info(f"Convert {N} samples...")

    tol = 0.0000001

    folder = (Path(args.path).stem + "-samples").replace("stan_fit_", "")
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    e_ref = None
    if args.ref_emb is not None:
        LOGGER.info(f"Load ref embedding from {args.ref_emb} for the vocabulary...")
        e_ref = Embedding(saved_model_path=args.ref_emb)
        words = [wd for wd in e_ref.vocabulary if "_c" not in wd]
        words = sorted(words)
        contexts = [wd + "_c" for wd in words]
        vocab = bidict.bidict({ix: wd for ix, wd in enumerate(words)})

    for n in tqdm.tqdm(list(range(N))):
        e_sample = Embedding(set(words), dimensionality=K)
        theta_n = fit.variational_sample[n, 3:]
        LOGGER.debug(f"theta_n shape {theta_n.shape} for round {n}")
        theta_n = np.reshape(theta_n, (K, 2 * V))
        #print(theta_n.shape)
        #print(e_sample[words + contexts].shape)
        e_sample[words + contexts] = theta_n.T
        #print(fit.variational_sample[n][:5])
        #print(e_sample[vocab[1]][0])
        error = np.abs(fit.variational_sample[n][4] - e_sample[vocab.inv[1]][0])
        assert error < tol, f"{fit.variational_sample[n][4]} {e_sample[vocab[1]][0]}"

        e_sample.save(str((folder / f"sample-{n}.pkl").absolute()))
        


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--vocab", type=str, default=None)
    argparser.add_argument("--ref_emb", type=str, default=None)
    #argparser.add_argument("--outpath", type=str, default=None)
    #argparser.add_argument("--format", type=str, default="json", choices=['json', 'embedding'])
    args = argparser.parse_args()
    print(args)
    main(args)
