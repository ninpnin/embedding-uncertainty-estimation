import unittest

#from embedding_uncertainty import cbow_gibbs
from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import networkx as nx
import progressbar
import math
import json
import os


class Test(unittest.TestCase):

    def test_cbow(self):
        """
        Test CBOW sampling with per-word alpha sampling
        """

        N, V, K = 1000, 100, 10
        vocab = set()
        data = []
        datafile = f'data/data-cbow-K-{K}-V-{V}-N-100000.json'
        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), datafile)
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        # Aggregate data into matrices
        w = []
        C = []
        x = []
        for elem in d[:N]:
            w_i, C_i, x_i = elem["w"], elem["C"], elem["x"]
            w.append(w_i)
            C.append(C_i)
            x.append(x_i)

        w = tf.constant(w)
        C = tf.constant(C)
        x = tf.constant(x)

        for elem in d:
            w_i, C_i = elem["w"], elem["C"]
            vocab.add(w_i)
            for v in C_i:
                vocab.add(v + "_c")

        e = Embedding(vocab, dimensionality=K, lambda0=1.0)

    def test_cbow_parallel(self):
        """
        Test CBOW sampling with parallellized alpha sampling
        """
        N, V, K = 1000, 100, 10
        vocab = set()
        data = []
        datafile = f'data/data-cbow-K-{K}-V-{V}-N-100000.json'
        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), datafile)
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        # Aggregate data into matrices
        w = []
        C = []
        x = []
        for elem in d[:N]:
            w_i, C_i, x_i = elem["w"], elem["C"], elem["x"]
            w.append(w_i)
            C.append(C_i)
            x.append(x_i)

        w = tf.constant(w)
        C = tf.constant(C)
        x = tf.constant(x)

        for elem in d:
            w_i, C_i = elem["w"], elem["C"]
            vocab.add(w_i)
            for v in C_i:
                vocab.add(v + "_c")

        e = Embedding(vocab, dimensionality=K, lambda0=1.0)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
