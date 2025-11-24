from embedding_uncertainty import embedding_gibbs
from embedding_uncertainty import embedding_gibbs_tf, embedding_gibbs_tf_fast
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
    parser.add_argument("--shuffle_data", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--map_estimate", type=str, default=None)
    parser.add_argument("--freeze_params", type=str, nargs="+", default=None)
    parser.add_argument("--lambda0", type=float, default=None, help="Prior strength (variance). If not specified, set to K")
    parser.add_argument("--example_word", type=str, default="word0")
    parser.add_argument("--use_tf", type=bool, default=False)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--pg_iter", type=int, default=5)
    parser.add_argument("--mvn_method", type=str, default="cholesky", choices=["cholesky", "svd"])
    parser.add_argument("--calculate_p", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
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

    lambda0 = args.lambda0
    if lambda0 is None:
        lambda0 = float(args.dim)
    e = Embedding(vocab, dimensionality=args.dim, lambda0=lambda0)
    freeze_params = []

    if args.map_estimate is not None:

        #freeze_params = [f"word{i}_c" for i in range(e.dimensionality)]
        freeze_params = [wd for wd in sorted(list(e.vocabulary)) if "_c" in wd]
        freeze_params = freeze_params[:e.dimensionality]

        if args.freeze_params is not None:
            if len(args.freeze_params) != e.dimensionality:
                LOGGER.error(f"Number of parameters to be frozen {len(args.freeze_params)} does not match dimensionality {e.dimensionality}")

            freeze_params = args.freeze_params
            if freeze_params[0][-3:] != "_c":
                LOGGER.warning(f"Parameters to be frozen provided as words without _c; adding it...")
                freeze_params = [wd.split("_")[0] + "_c" for wd in freeze_params]

                for wd in freeze_params:
                    if wd not in e:
                        LOGGER.critical(f"Frozen parameter '{wd}' not in vocabulary!")
                        exit()

        LOGGER.train(f"Copy from MAP and freeze following params: {freeze_params}")
        e_map = Embedding(saved_model_path=args.map_estimate)
        LOGGER.train(f"e {e[freeze_params].shape} emap { e_map[freeze_params].shape}")
        LOGGER.train(f"e {e[freeze_params].dtype} emap { e_map[freeze_params].dtype}")
        e[freeze_params] = e_map[freeze_params]

    WARMUP = args.samples // 2
    p_avg = None
    pathstem = Path(args.datapath).stem.replace("_", "-")
    V = len(vocab)
    # Generate a random string to make runs pseudo unique
    randomchars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    samples_folder = f"{pathstem}-gibbs-N-{args.data_len}-K-{args.dim}-V-{V}-PG-{args.pg_iter}-{args.prefix}-{randomchars}"

    gibbs_generator = embedding_gibbs_tf_fast(e, data, rounds=args.samples, plot=args.plot, polyagamma_iter=args.pg_iter, ll_every=100)
    for sample_ix, e_sample in enumerate(gibbs_generator):
        word0sample = e_sample[args.example_word].numpy()
        LOGGER.info(f"Example word {args.example_word}: {e_sample[args.example_word]}")

    gibbs_generator = embedding_gibbs_tf(e, data, rounds=args.samples, plot=args.plot, polyagamma_iter=args.pg_iter, ll_every=100)
    for sample_ix, e_sample in enumerate(gibbs_generator):
        word0sample = e_sample[args.example_word].numpy()
        LOGGER.info(f"Example word {args.example_word}: {e_sample[args.example_word]}")
