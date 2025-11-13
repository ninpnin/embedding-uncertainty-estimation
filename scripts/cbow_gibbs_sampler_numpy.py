import numpy as np
from polyagamma import random_polyagamma
#import os
import json
#import tensorflow as tf
import tqdm
from probabilistic_word_embeddings.embeddings import Embedding
import bidict
import random, string
from trainerlog import get_logger
LOGGER = get_logger("cbow-numpy-gibbs")
from pathlib import Path
from embedding_uncertainty import cbow_gibbs_sampler

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="tests/data/data-cbow-K-10-V-100-N-100000.json")
    parser.add_argument("--K", type=int, default=10, help="Dimensionality of the embeddings")
    parser.add_argument("--S", type=int, default=2, help="S hyperparameter for the PG-Gibbs algorithm")
    parser.add_argument("--data_len", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--lambda0", type=float, default=None, help="Prior strength (variance). If not specified, set to K")
    parser.add_argument("--results_folder", type=str, default="results", help="Where the samples folder should be placed")
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")


    N, K = args.data_len, args.K
    #N, V, K = 1000, 100, 10
    #datafile = f'tests/data/data-cbow-K-{K}-V-{V}-N-100000.json'
    #TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), datafile)
    with open(args.datapath) as f:
        docs = json.load(f)

    # --- Data setup ---

    vocab = set()
    # Aggregate data into matrices
    ww = []
    CC = []
    xx = []
    for elem in docs[:N]:
        w_i, C_i, x_i = elem["w"], elem["C"], elem["x"]
        ww.append(w_i)
        CC.append(C_i)
        xx.append(x_i)

    vocab = set(ww)
    V = len(vocab)

    word2id = bidict.bidict({wd: ix for ix, wd in enumerate(vocab)})

    w_idx = np.array([word2id[w] for w in ww], dtype=int)
    C_idx = np.array([[word2id[v] for v in C_i] for C_i in CC])
    x_vec = np.array(xx, dtype=float)

    # --- SAMPLER SETUP ----
    lam = K
    if args.lambda0 is not None:
        lam = args.lambda0
        LOGGER.info(f"Set lambda0 from argparse parameters {lam}")
    else:
        LOGGER.info(f"Set lambda0 to default sqrt(K) = {lam}")

    
    n_samples = args.n_samples

    rho_samples, alpha_samples = cbow_gibbs_sampler(w_idx, C_idx, x_vec,
                                    n_samples=n_samples, S=args.S, V=V, K=K, lam=lam)

    pathstem = Path(args.datapath).stem.replace("_", "-")
    randomchars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    samples_folder = f"{pathstem}-cbow-gibbs-numpy-N-{N}-K-{K}-V-{V}-{randomchars}"
    results_folder = Path(args.results_folder)
    results_folder.mkdir(exist_ok=True)
    samples_folder = (results_folder / samples_folder)
    LOGGER.info(f"Mkdir {samples_folder} ...")
    samples_folder.mkdir(exist_ok=True)

    words = [word2id.inv[ix] for ix in range(V)]
    contexts = [wd + "_c" for wd in words]
    for sample_ix, emb in tqdm.tqdm(enumerate(zip(rho_samples, alpha_samples))):
        rho, alpha = emb
        e_sample = Embedding(set(vocab), dimensionality=K, lambda0=lam)
        e_sample[words] = rho
        e_sample[contexts] = alpha

        sample_path = samples_folder / f"sample-{sample_ix}.json"
        sample_path_str = str(sample_path.resolve())
        e_sample.save(sample_path_str)
