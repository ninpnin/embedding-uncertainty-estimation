from embedding_uncertainty import embedding_gibbs
from embedding_uncertainty import embedding_gibbs_tf
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
    parser.add_argument("--pg_iter", type=int, default=50)
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

    LOGGER.info(f"Make folder {samples_folder} ...")
    Path(samples_folder).mkdir(exist_ok=True)

    gibbs_generator = embedding_gibbs(e, data, rounds=args.samples, polyagamma_iter=args.pg_iter, freeze_params=freeze_params, plot=args.plot)
    if args.use_tf:
        gibbs_generator = embedding_gibbs_tf(e, data, rounds=args.samples, polyagamma_iter=args.pg_iter, freeze_params=freeze_params, multivariate_method=args.mvn_method, plot=args.plot)
    for sample_ix, e_sample in enumerate(gibbs_generator):
        word0sample = e_sample[args.example_word].numpy()
        LOGGER.info(f"Example word {args.example_word}: {e_sample[args.example_word]}")
        x.append(word0sample[0])
        y.append(word0sample[1])

        if args.calculate_p:
            rho = e_sample[[wd for wd in e.vocabulary if "_c" not in wd]].numpy()
            alpha = e_sample[[wd for wd in e.vocabulary if "_c" in wd]].numpy()
            eta = rho @ alpha.T
            p = tf.math.sigmoid(eta)

            if sample_ix >= WARMUP:
                if p_avg is None:
                    p_avg = p
                else:
                    p_avg += p

        sample_path = f"{samples_folder}/sample-{sample_ix}.pkl"
        LOGGER.info(f"Save sample to {sample_path} ...")
        e_sample.save(sample_path)

    theta_true = np.array(d["theta"])
    rho_true = theta_true[:theta_true.shape[0] // 2]
    alpha_true = theta_true[theta_true.shape[0] // 2:]

    if args.calculate_p:
        p_true = tf.math.sigmoid(rho_true @ alpha_true.T).numpy()
        p_avg = p_avg / (args.samples - WARMUP)
        
        LOGGER.train(f"RMSE baseline {np.sqrt(np.mean((p_true - np.mean(p_true)) ** 2))}")
        RMSE = np.sqrt(np.mean((p_true - p_avg) ** 2))
        LOGGER.train(f"RMSE: {RMSE}")

    sns.set_theme()
    sns.lineplot(x=x, y=y, sort=False)
    plt.savefig(f"gibbsample-{args.example_word}-{args.samples}.png")
    plt.show()
