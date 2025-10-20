from embedding_uncertainty.laplace_approx import perword_laplace_approx
from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import copy
import json
import tqdm
import progressbar
import pandas as pd
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import random, string

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--shuffle_data", type=bool, default=False)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--map_estimate", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--mvn_method", type=str, default="cholesky", choices=["cholesky", "svd"])
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]

    vocab = set()
    if args.datapath is not None:
        with open(args.datapath, "rb") as f:
            d = json.load(f)

        LOGGER.debug(f"Keys: {d.keys()}")

        if args.shuffle_data:
            LOGGER.info(f"Shuffle data...")
            random.shuffle(d["data"])
        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            vocab.add(elem["w"])
            data.append((w,v,x))

    if args.data_len is not None:
        data = data[:args.data_len]

    x, y = [], []

    LOGGER.train(f"Load in MAP estimate")
    e_map = Embedding(saved_model_path=args.map_estimate)

    WARMUP = args.samples // 2
    p_avg = None
    pathstem = Path(args.datapath).stem.replace("_", "-")
    V = len(vocab)
    K = e_map.dimensionality
    # Generate a random string to make runs pseudo unique
    randomchars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    samples_folder = f"{pathstem}-perword-laplace-N-{args.data_len}-K-{K}-V-{V}-{args.prefix}-{randomchars}"

    LOGGER.info(f"Make folder {samples_folder} ...")
    Path(samples_folder).mkdir(exist_ok=True)

    for sample_ix in range(args.samples):
        e_sample = copy.deepcopy(e_map)
        wds = list(e_map.vocabulary)
        e_sample[wds] = e_sample[wds] * 0.0

        for wd in tqdm.tqdm(list(e_map.vocabulary)):
            e_wd = perword_laplace_approx(wd, e_map, data, sample_n=1)[0]
            #print(e_wd)
            e_sample[wd] = e_wd

        sample_path = f"{samples_folder}/sample-{sample_ix}.pkl"
        LOGGER.info(f"Save sample to {sample_path} ...")
        e_sample.save(sample_path)
