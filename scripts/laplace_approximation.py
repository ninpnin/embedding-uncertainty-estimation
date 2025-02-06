from probabilistic_word_embeddings.embeddings import Embedding
from embedding_uncertainty.laplace_approx import full_hessian, fixed_inverse_hessian, laplace_approx
from embedding_uncertainty.laplace_approx import laplace_approx_sigma
import numpy as np
import tensorflow as tf
import copy
import json
import progressbar
import pandas as pd
from trainerlog import get_logger
import bidict
from matplotlib import pyplot as plt
import polars as pl
from pathlib import Path

LOGGER = get_logger("laplace")

def cossim(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_p(e, words, contexts):
    P = sigmoid(e[words].numpy() @ e[contexts].numpy().T)
    #print("P shape", P.shape)
    return P

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, default="example_embedding.pkl")
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--word", type=str, default="word1")
    parser.add_argument("--context", type=str, default="word2_c")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--ci_alpha", type=float, default=0.1)
    parser.add_argument("--elementwise", type=bool, default=False)
    parser.add_argument("--save_folder", type=str, default=None)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    LOGGER.train(f"Load embedding from {args.embedding}...")
    e_map = Embedding(saved_model_path=args.embedding)
    sim = tf.math.sigmoid(tf.reduce_sum(e_map[args.word] * e_map[args.context]))
    LOGGER.debug(f"Sim (MAP) [{args.word} {args.context}] {sim}")
    if e_map.lambda0 is None:
        e_map.lambda0 = e_map.dimensionality
    LOGGER.debug(f"lambda0: {e_map.lambda0}")
    K = e_map.dimensionality
    V = len([wd for wd in e_map.vocabulary if "_c" not in wd])
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = []#('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]
    theta = None
    e_truth = None
    words, contexts = [], []
    if args.datapath is not None:
        with open(args.datapath, "rb") as f:
            d = json.load(f)

        LOGGER.debug(f"Keys: {d.keys()}")

        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            data.append((w,v,x))

        theta = np.array(d["theta"])
        LOGGER.info(f"theta: {theta.shape}")
        e_truth = copy.deepcopy(e_map)
        words = sorted([wd for wd in e_truth.vocabulary if "_c" not in wd])
        contexts = [wd + "_c" for wd in words] 
        e_truth[words + contexts] = theta

    if args.data_len is not None:
        data = data[:args.data_len]

    p_truth = get_p(e_truth, words, contexts)
    LOGGER.debug(f"p_truth: {p_truth.shape} {p_truth[:5,:5]}")
    p_map = get_p(e_map, words, contexts)
    LOGGER.debug(f"p_map: {p_map.shape} {p_map[:5,:5]}")

    RMSE = np.sqrt(np.mean((p_map - p_truth) ** 2))
    LOGGER.info(f"RMSE: {RMSE}")
    RMSE_baseline = np.sqrt(np.mean((p_truth - 0.5) ** 2))
    LOGGER.info(f"RMSE_baseline: {RMSE_baseline}")

    p_samples = []
    sample_ix = 0
    save_folder = None
    if args.save_folder is not None:
        save_folder = Path(args.save_folder)
        save_folder.mkdir(exist_ok=True)
    for e_sample in laplace_approx(e_map, data, samples=args.samples, rotational_fix=True):
        if args.save_folder is not None:
            e_sample.save(str((save_folder / f"sample-{sample_ix}.pkl").absolute()))
        p_sample = get_p(e_sample, words, contexts)
        p_samples.append(p_sample)
        sample_ix += 1

    p_samples = np.array(p_samples)
    LOGGER.info(f"p_samples: {p_samples.shape}")
    
    lower = np.quantile(p_samples, args.ci_alpha / 2.0, axis=0)
    upper = np.quantile(p_samples, 1.0 - args.ci_alpha / 2.0, axis=0)

    #print(lower)
    #print(p_truth)
    #print(upper)

    within_CI = (lower < p_truth) * (upper > p_truth)
    coverage = np.mean(within_CI)
    LOGGER.info(f"Coverage: {coverage}")

    map_stem = Path(args.embedding).stem
    results_df = pl.DataFrame({"K": K, "V": V, "N": args.data_len, "map": map_stem, "coverage": coverage, "RMSE": RMSE, "RMSE_normalized": RMSE / RMSE_baseline})
    print(results_df)

    results_path = Path("logs") / "laplace-results.csv"
    if results_path.exists():
        old_results = pl.read_csv(results_path)
        results_df = pl.concat([old_results, results_df])

    results_df = results_df.unique(["K", "V", "map", "N"])
    results_df = results_df.sort("K", "V", "N", "map")

    results_df.write_csv(results_path)
    if args.elementwise:
        Sigma = laplace_approx_sigma(e_map, data, samples=args.samples, rotational_fix=True)

        print("Sigma shape", Sigma.shape)
        trues = 0
        falses = 0
        for word in words:
            for context in contexts:
                i, j = e_map.vocabulary[word], e_map.vocabulary[context]
                #print(i,j)
                ik, jk = i * K, j * K
                Sigma_ii = Sigma[ik:ik+K,ik:ik+K]
                Sigma_jj = Sigma[jk:jk+K,jk:jk+K]
                Sigma_ij = Sigma[ik:ik+K,jk:jk+K]

                #print(Sigma_ii)
                #print(Sigma_jj)
                #print(Sigma_ij)

                Sigma_wv = np.zeros((2*K, 2*K))
                Sigma_wv[:K, :K] = Sigma_ii
                Sigma_wv[K:, K:] = Sigma_jj
                Sigma_wv[:K, K:] = Sigma_ij
                Sigma_wv[K:, :K] = Sigma_ij.T

                L = None
                L = np.linalg.cholesky(Sigma_wv)

                deviation = L @ np.random.randn(2*K, args.samples)
                #print(deviation.shape)
                #print(Sigma_wv)
                dev_rhos = deviation[:K].T
                dev_alphas = deviation[K:].T
                #print()
                rhos = e_map[word].numpy() + dev_rhos
                alphas = e_map[context].numpy() + dev_alphas

                #print((rhos * alphas).shape)
                etas = np.sum(rhos * alphas, axis=1)
                p_wv = sigmoid(etas)
                lower, upper = np.quantile(p_wv, args.ci_alpha / 2.0), np.quantile(p_wv, 1.0 - args.ci_alpha / 2.0)
                #print(p_wv)
                #print(lower, p_truth[i, j - V], upper)
                #print(p_map[i, j - V])
                if lower < p_truth[i, j - V] and upper > p_truth[i, j - V]:
                    #print("True")
                    trues += 1
                else:
                    #print("false")
                    falses += 1

        print("ratio", trues / (trues + falses))

                #exit()




