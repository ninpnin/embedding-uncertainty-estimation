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

def generate_e(K, V, epsilon=1.0):
    vocabulary = set([f"word{ix}" for ix in range(V)])
    e = Embedding(vocabulary, dimensionality=K)
    x = np.random.randn(2 * V,K)
    e.theta.assign(np.array(epsilon * x / np.sqrt(K)))
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
    N = 100000
    K, V = 10, 100
    e, vocabulary = generate_e(K, V)

    data = []
    for ix in tqdm.tqdm(range(N)):
        datapoint = generate_data_point(e, vocabulary)
        data.append(datapoint)
        #print(datapoint)
        #exit()

    json_output = json.dumps(data, indent=2)
    #print(json_output)

    e_filename = f"tests/data/e_cbow-K-{K}-V-{V}.json"
    e.save(e_filename)
    data_filename = f"tests/data/data-cbow-K-{K}-V-{V}-N-{N}.json"
    print(data_filename)
    with open(data_filename, "w") as f:
        f.write(json_output)
