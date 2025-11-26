import unittest

from embedding_uncertainty import embedding_gibbs, embedding_gibbs_tf
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
import numpy as np
import tensorflow as tf
import networkx as nx
import progressbar
import math
import json
import os

def get_p_matrix(e):
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    words = sorted([wd for wd in e.vocabulary if "_c" not in wd])
    contexts = sorted([wd for wd in e.vocabulary if "_c" in wd])

    alpha = e[contexts].numpy()
    rho = e[contexts].numpy()

    return sigmoid(alpha.T @ rho)

class Test(unittest.TestCase):

    def test_laplacian(self):
        """
        Check that aggregating the sufficient statistics yields the same results as not doing that
        """

        vocab = set()
        data = []
        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data/testdata.json')
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        for elem in d:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            vocab.add(elem["w"])
            data.append((w,v,x))

        g = nx.Graph()
        g.add_edge("word1", "word2")
        e_laplacian = LaplacianEmbedding(vocab, dimensionality=3, graph=g, lambda1=0.001, lambda0=1.0)
        e_reference = Embedding(vocab, dimensionality=3, lambda0=1.0)

        ROUNDS = 1000

        reference_generator = embedding_gibbs_tf(e_reference, data, rounds=ROUNDS, plot=False)
        laplacian_generator = embedding_gibbs_tf(e_laplacian, data, rounds=ROUNDS, plot=False)

        laplacian_ps = []
        for _, e_sample in enumerate(laplacian_generator):
            p = get_p_matrix(e_sample)
            laplacian_ps.append(p)

        laplacian_ps = np.array(laplacian_ps)

        reference_ps = []
        for _, e_sample in enumerate(reference_generator):
            p = get_p_matrix(e_sample)
            reference_ps.append(p)
        reference_ps = np.array(reference_ps)

        bar_agg = np.mean(laplacian_ps, axis=0)
        bar_ref = np.mean(reference_ps, axis=0)
        print(bar_agg)
        print(bar_ref)

        diff = np.abs(bar_ref - bar_agg)
        MAE = np.mean(diff)

        self.assertLessEqual(MAE, 0.025, f"MAE should be less than 0.025, was {MAE}")
        self.assertNotEqual(MAE, 0.00, f"MAE not be 0.0")



    def test_strong_laplacian(self):
        """
        Check that aggregating the sufficient statistics yields the same results as not doing that
        """

        vocab = set()
        data = []
        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data/testdata.json')
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        for elem in d:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            vocab.add(elem["w"])
            data.append((w,v,x))

        g = nx.Graph()
        g.add_edge("word1", "word2")
        e_laplacian = LaplacianEmbedding(vocab, dimensionality=3, graph=g, lambda1=1000.0, lambda0=1.0)
        e_reference = Embedding(vocab, dimensionality=3, lambda0=1.0)

        ROUNDS = 1000

        reference_generator = embedding_gibbs_tf(e_reference, data, rounds=ROUNDS, plot=False)
        laplacian_generator = embedding_gibbs_tf(e_laplacian, data, rounds=ROUNDS, plot=False)

        laplacian_ps = []
        dist = 0.0
        for _, e_sample in enumerate(laplacian_generator):
            dist = np.sum( (e_sample["word1"]- e_sample["word2"]) ** 2)
            p = get_p_matrix(e_sample)
            laplacian_ps.append(p)

        self.assertLessEqual(dist, 0.1, f"MAE should be less than 0.025, was {dist}")


    def test_laplacian_no_data(self):
        """
        Check that aggregating the sufficient statistics yields the same results as not doing that
        """

        vocab = set()
        data = []
        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data/testdata.json')
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        data = []
        for elem in d:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            vocab.add(elem["w"])
            if elem["w"] not in ["word1", "word2"]:
                if elem["v"] not in ["word1", "word2"]:
                    data.append((w,v,x))

        g = nx.Graph()
        g.add_edge("word1", "word2")
        e_laplacian = LaplacianEmbedding(vocab, dimensionality=3, graph=g, lambda1=1.0, lambda0=1.0)

        ROUNDS = 1000
        laplacian_generator = embedding_gibbs_tf(e_laplacian, data, rounds=ROUNDS, plot=False)
        
        word1 = []
        word2 = []
        for _, e_sample in enumerate(laplacian_generator):
            word1.append(e_sample["word1"])
            word2.append(e_sample["word2"])

        word1 = np.array(word1)
        word2 = np.array(word2)
        print(word1)
        print(word1.shape)
        # word1mean = word

        x = word1[:,0]
        y = word2[:,0]

        P = np.identity(2) + np.array([[1, -1], [-1, 1]])
        covariance_matrix_truth = np.linalg.inv(P)

        assert len(x) == ROUNDS
        covariance_matrix = np.cov(x, y)
        var1 = covariance_matrix[0,0]
        var2 = covariance_matrix[1,1]
        cov = covariance_matrix[0,1]
        print(covariance_matrix)
        print(covariance_matrix_truth)
        var_truth = covariance_matrix_truth[0,0]
        cov_truth = covariance_matrix_truth[0,1]
        self.assertAlmostEqual(var1, var_truth, 1, f"Var should be roughly than {var_truth}, was {var1}")
        self.assertAlmostEqual(var2, var_truth, 1, f"Var should be roughly than {var_truth}, was {var2}")
        self.assertAlmostEqual(cov, cov_truth, 1, f"Cov should be roughly than {cov_truth}, was {cov}")
        #self.assertLessEqual(dist, 0.1, f"MAE should be less than 0.025, was {dist}")


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
