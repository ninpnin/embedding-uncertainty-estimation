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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--lambda0", type=float, default=None, help="Prior strength (variance). If not specified, set to K")
    parser.add_argument("--example_word", type=str, default="word0")
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]

    vocab = set()
    data = None
    if args.datapath is not None:
        with open(args.datapath) as f:
            data = json.load(f)

    for elem in data:
        w_i, C_i = elem["w"], elem["C"]
        vocab.add(w_i)
        for v in C_i:
            vocab.add(v)

    if args.data_len is not None:
        data = data[:args.data_len]


    lambda0 = args.lambda0
    if lambda0 is None:
        lambda0 = float(args.dim)

    e = Embedding(vocab, dimensionality=args.dim, lambda0=lambda0)
    freeze_params = []

    WARMUP = args.samples // 2
    p_avg = None
    pathstem = Path(args.datapath).stem.replace("_", "-")
    V = len(vocab)
    # Generate a random string to make runs pseudo unique
    randomchars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    samples_folder = f"{pathstem}-cbow-gibbs-N-{args.data_len}-K-{args.dim}-V-{V}-{args.prefix}-{randomchars}"


    gibbs_generator = cbow_gibbs_parallellized(e, data, rounds=args.samples)
    local = None
    for sample_ix, e_sample in enumerate(gibbs_generator):
        LOGGER.info(f"Make folder {samples_folder} ...")
        #Path(samples_folder).mkdir(exist_ok=True)
        word0sample = e_sample[args.example_word].numpy()
        LOGGER.info(f"Example word {args.example_word}: {e_sample[args.example_word]}")

        