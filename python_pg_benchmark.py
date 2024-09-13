from embedding_uncertainty import embedding_gibbs, polyagamma_sampler
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
LOGGER.info("Done")
import progressbar
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


from time import sleep, perf_counter as pc
t0 = pc()
sleep(1)
print(pc()-t0)

def main():
    dim = 100
    N = 2300
    r = 1000

    beta_true = np.random.randn(dim) / np.sqrt(dim)
    X = np.random.randn(dim, N)
    eta = tf.math.sigmoid(X.T @ beta_true).numpy()
    print(eta.shape)
    beta_init = np.random.randn(dim) / np.sqrt(dim)
    y = np.random.binomial(1, eta)
    print(y)

    time_0 = pc()
    for beta_hat in polyagamma_sampler(beta_init, X.T, y, iterations=r):
        pass
    time_1 = pc()
    timedelta = time_1 - time_0
    print("Seconds per iteration", timedelta / r)
    print("Iterations per second", 1.0/ (timedelta / r ))

    iterations_per_word = 50
    print("Words per second", 1.0/ (iterations_per_word* timedelta / r ), "(at", iterations_per_word, "iterations per word)")

    print("Observations per second", N/ (iterations_per_word* timedelta / r ), "(at", iterations_per_word, "iterations per word)")

def polyagamma_fun(x):
    beta_init, X, y = x
    return list(polyagamma_sampler(beta_init, X.T, y, iterations=100))[-1]

def main_parallel():
    dim = 100
    N = 2300
    r = 100
    parallel_threads = 1000

    beta_true = np.random.randn(dim) / np.sqrt(dim)
    X = np.random.randn(dim, N)
    eta = tf.math.sigmoid(X.T @ beta_true).numpy()
    beta_init = np.random.randn(dim) / np.sqrt(dim)
    y = np.random.binomial(1, eta)
    #def polyagamma_fun(beta_init, X, y):
    #    return list(polyagamma_sampler(beta_init, X.T, y, iterations=r))[-1]

    unknowns = []
    params = [(beta_init, X, y)] * parallel_threads

    sample = polyagamma_fun(params[0])
    print(sample)
    time_0 = pc()
    pool = Pool()
    for unk in tqdm(pool.imap(polyagamma_fun, params), total=len(params)):
        unknowns.append(unk)

    time_1 = pc()
    timedelta = time_1 - time_0
    print("Seconds per iteration", timedelta / (r*parallel_threads))
    print("Iterations per second", parallel_threads/ (timedelta / r ))

    iterations_per_word = 50
    print("Words per second", parallel_threads/ (iterations_per_word* timedelta / r ), "(at", iterations_per_word, "iterations per word)")

    print("Observations per second", parallel_threads*N/ (iterations_per_word* timedelta / r ), "(at", iterations_per_word, "iterations per word)")

if __name__ == '__main__':
    #main()
    main_parallel()