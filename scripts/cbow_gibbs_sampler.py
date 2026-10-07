from embedding_uncertainty import cbow_gibbs_parallellized
from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import copy
import json
import progressbar
import pandas as pd
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import random, string
import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--lambda0", type=float, default=None, help="Prior strength (variance). If not specified, set to K")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--example_word", type=str, default="word0")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--results_folder", type=str, default="results", help="Where the samples folder should be placed")
    parser.add_argument("--calculate_p", type=bool, default=False, help="Calculate alpha rho.T for ESS etc.")
    parser.add_argument("--benchmark", type=bool, default=False, help="Only run the script; don't save embeddings")
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]

    vocab = set()
    data = None
    if args.datapath is not None:
        with open(args.datapath) as f:
            data = json.load(f)
            if not isinstance(data, list) and "data" in data.keys():
                LOGGER.warn("JSON is nested; load contents of 'data' variable")
                data = data["data"]

    for elem in data:
        w_i, C_i = elem["w"], elem["C"]
        vocab.add(w_i)
        for v in C_i:
            vocab.add(v)

    if args.data_len is not None:
        data = data[:args.data_len]

    K = args.dim
    lam = K
    if args.lambda0 is not None:
        lam = args.lambda0
        LOGGER.info(f"Set lambda0 from argparse parameters {lam}")
    else:
        LOGGER.info(f"Set lambda0 to default sqrt(K) = {lam}")

    e = Embedding(vocab, dimensionality=args.dim, lambda0=lam)
    freeze_params = []

    WARMUP = args.samples // 2
    p_avg = None
    pathstem = Path(args.datapath).stem.replace("_", "-")
    V = len(vocab)
    # Generate a random string to make runs pseudo unique
    randomchars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    samples_folder = f"{pathstem}-cbow-gibbs-N-{args.data_len}-K-{K}-V-{V}-{args.prefix}-{randomchars}"
    results_folder = Path(args.results_folder)
    results_folder.mkdir(exist_ok=True)
    samples_folder = (results_folder / samples_folder)
    LOGGER.info(f"Mkdir {samples_folder} ...")
    samples_folder.mkdir(exist_ok=True)

    gibbs_generator = cbow_gibbs_parallellized(e, data, rounds=args.samples, batch_size=args.batch_size)
    local = None

    cossim, words = None, list(vocab)
    if args.calculate_p:
        cossim = np.zeros((args.samples // 2, V,V))

    for sample_ix, e_sample in enumerate(gibbs_generator):
        Path(samples_folder).mkdir(exist_ok=True)
        word0sample = e_sample[args.example_word].numpy()
        LOGGER.info(f"Example word {args.example_word}: {e_sample[args.example_word]}")

        sample_path = samples_folder / f"sample-{sample_ix}.json"
        sample_path_str = str(sample_path.resolve())
        if not args.benchmark:
            e_sample.save(sample_path_str)

        if args.calculate_p and sample_ix >= args.samples // 2:
            cossim_ix = e[words].numpy()
            rho_norm = np.linalg.norm(cossim_ix, axis=1)
            cossim_ix = ((cossim_ix.T) / rho_norm).T
            cossim_ix = cossim_ix @ cossim_ix.T
            cossim[sample_ix - args.samples // 2] = cossim_ix

    LOGGER.info("Calculate ESS based on cossim")
    if args.calculate_p:
        import arviz as az
        ess_arr_cos = []
        for w in tqdm.tqdm(range(V)):
            for v in range(V):
                cossim_samples = cossim[:, w, v]
                ESS_cos = az.ess(cossim_samples)
                ess_arr_cos.append(ESS_cos)
        
        ess_arr_cos = np.array(ess_arr_cos)
        LOGGER.train(f"ESS (cossim): {np.mean(ess_arr_cos)} (+- {np.std(ess_arr_cos)})")
