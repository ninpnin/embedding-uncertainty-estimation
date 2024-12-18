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
import random

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--map_estimate", type=str, default=None)
    parser.add_argument("--lambda0", type=float, default=None, help="Prior strength (variance)")
    parser.add_argument("--example_word", type=str, default="word0")
    parser.add_argument("--use_tf", type=bool, default=False)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--pg_iter", type=int, default=50)
    parser.add_argument("--mvn_method", type=str, default="svd")
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]

    vocab = set()
    if args.datapath is not None:
        with open(args.datapath, "rb") as f:
            d = json.load(f)

        LOGGER.debug(f"Keys: {d.keys()}")

        random.shuffle(d["data"])
        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            vocab.add(elem["w"])
            data.append((w,v,x))

    if args.data_len is not None:
        data = data[:args.data_len]

    x, y = [], []

    e = Embedding(vocab, dimensionality=args.dim)
    freeze_params = []

    if args.map_estimate is not None:

        #freeze_params = [f"word{i}_c" for i in range(e.dimensionality)]
        freeze_params = [wd for wd in sorted(list(e.vocabulary)) if "_c" in wd]
        freeze_params = freeze_params[:e.dimensionality]
        LOGGER.train(f"Copy from MAP and freeze following params {freeze_params}")
        e_map = Embedding(saved_model_path=args.map_estimate)
        LOGGER.train(f"e {e[freeze_params].shape} emap { e_map[freeze_params].shape}")
        LOGGER.train(f"e {e[freeze_params].dtype} emap { e_map[freeze_params].dtype}")
        e[freeze_params] = e_map[freeze_params]

    rows = []
    columns = sorted(list(e.vocabulary))

    WARMUP = args.samples // 2
    p_avg = None
    pathstem = Path(args.datapath).stem.replace("_", "-")
    gibbs_generator = embedding_gibbs(e, data, rounds=args.samples, polyagamma_iter=args.pg_iter, lambda0=args.lambda0, freeze_params=freeze_params)
    if args.use_tf:
        gibbs_generator = embedding_gibbs_tf(e, data, rounds=args.samples, polyagamma_iter=args.pg_iter, lambda0=args.lambda0, freeze_params=freeze_params, multivariate_method=args.mvn_method)
    for sample_ix, e_sample in enumerate(gibbs_generator):
        print(e_sample)
        word0sample = e_sample[args.example_word].numpy()
        print(e_sample[args.example_word])
        x.append(word0sample[0])
        y.append(word0sample[1])

        vectors = e_sample[columns].numpy()
        print(vectors.shape)

        newcols, row = [], []
        for dim in range(vectors.shape[-1]):
            newcols += [f"{wd}_{dim}" for wd in columns]
            row += [vectors[ix][dim] for ix, _ in enumerate(columns)]
        rows.append(row)

        rho = e_sample[[wd for wd in columns if "_c" not in wd]].numpy()
        alpha = e_sample[[wd for wd in columns if "_c" in wd]].numpy()
        eta = rho @ alpha.T
        p = tf.math.sigmoid(eta)

        if sample_ix >= WARMUP:
            if p_avg is None:
                p_avg = p
            else:
                p_avg += p

        if sample_ix % 100 == 0:
            df = pd.DataFrame(rows, columns=newcols)
            df = df[sorted(newcols)]
            print(df)
            Path()
            df.to_csv(f"gibbs-samples-N-{args.data_len}-D-{args.dim}-{args.prefix}{pathstem}.csv", index=False)
    
    theta_true = np.array(d["theta"])
    rho_true = theta_true[:theta_true.shape[0] // 2]
    alpha_true = theta_true[theta_true.shape[0] // 2:]

    p_true = tf.math.sigmoid(rho_true @ alpha_true.T).numpy()
    p_avg = p_avg / (args.samples - WARMUP)
    
    LOGGER.train(f"RMSE baseline {np.sqrt(np.mean((p_true - np.mean(p_true)) ** 2))}")
    RMSE = np.sqrt(np.mean((p_true - p_avg) ** 2))
    LOGGER.train(f"RMSE: {RMSE}")

    sns.set_theme()
    sns.lineplot(x=x, y=y, sort=False)
    plt.savefig(f"gibbsample-{args.example_word}-{args.samples}.png")
    plt.show()
