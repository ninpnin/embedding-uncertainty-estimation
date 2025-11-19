from trainerlog import get_logger
LOGGER = get_logger("cbow-sim")
LOGGER.info("Load modules..")
from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import networkx as nx
import progressbar
import math
import json
import os
import tqdm

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_e(K, V, epsilon=1.0, method="new"):
    if method not in ["new", "old"]:
        LOGGER.error(f"Unrecognized embedding generation method: {method}")
        return

    vocabulary = set([f"word{ix}" for ix in range(V)])
    e = Embedding(vocabulary, dimensionality=K)
    x = np.random.randn(2 * V,K)
    multiplier = np.sqrt(epsilon) / np.sqrt(np.sqrt(K))
    if method == "old":
        multiplier = np.sqrt(epsilon) / np.sqrt(K)

    LOGGER.info(f"Multiplier : {multiplier:.3f} using method: {method}")

    e.theta.assign(np.array(multiplier * x))

    words = [wd for wd in vocabulary]
    contexts = [f"{wd}_c" for wd in words]
    eta = e[contexts].numpy() @ e[words].numpy().T

    if eta.shape != (V, V):
        LOGGER.error(f"eta shape : {eta.shape} [{V}, {V}]")
        return
    else:
        LOGGER.info(f"eta shape : {eta.shape} [{V}, {V}]")
    LOGGER.info(f"eta st. dev. : {np.std(eta):.3f} [{epsilon}]")

    return e, vocabulary

def generate_data_point(e, vocabulary, ws=2):
    w = str(np.random.choice(list(vocabulary)))
    C = list(np.random.choice(list(vocabulary), size=ws))
    assert "_c" not in " ".join(C)

    rho = e[w]
    alpha = np.sum(e[[wd + "_c" for wd in C]], axis=0)
    eta = np.dot(rho, alpha)
    p = sigmoid(eta)
    x = int(np.random.rand() < p)

    datapoint = {"w": w, "C": C, "x": x}
    return datapoint

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10, help="Embedding dimensionality")
    parser.add_argument("--V", type=int, default=10, help="Embedding vocab size")
    parser.add_argument("--N", type=int, default=500_000, help="Observations to be simulated")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Embedding signal to noise ratio")
    parser.add_argument("--emb_sim_method", type=str, default="new")
    parser.add_argument("--e_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    N, K, V = args.N, args.K, args.V
    e, vocabulary = generate_e(K, V, epsilon=args.epsilon, method=args.emb_sim_method)

    data = []
    for ix in tqdm.tqdm(range(N)):
        datapoint = generate_data_point(e, vocabulary)
        data.append(datapoint)

    json_output = json.dumps(data, indent=2)

    e_filename = f"tests/data/e_cbow-K-{K}-V-{V}.json"
    if args.e_path is not None:
        e_filename = args.e_path
    LOGGER.info(f"Save embedding to: {e_filename} ..")
    e.save(e_filename)

    data_filename = f"tests/data/data-cbow-K-{K}-V-{V}-N-{N}.json"
    if args.data_path is not None:
        data_filename = args.data_path

    LOGGER.info(f"Save data to: {data_filename} ..")
    with open(data_filename, "w") as f:
        f.write(json_output)
