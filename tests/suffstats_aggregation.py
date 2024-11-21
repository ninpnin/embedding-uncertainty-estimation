import unittest

from embedding_uncertainty import embedding_gibbs
from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
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


    def test_sufficient_stats_aggregation(self):
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

        e_aggregated = Embedding(vocab, dimensionality=3)
        e_reference = Embedding(vocab, dimensionality=3)

        ROUNDS = 1000

        aggregated_generator = embedding_gibbs(e_aggregated, data, rounds=ROUNDS, aggregate=True, plot=False)
        reference_generator = embedding_gibbs(e_reference, data, rounds=ROUNDS, aggregate=False, plot=False)

        aggregated_ps = []
        for _, e_sample in enumerate(aggregated_generator):
            p = get_p_matrix(e_sample)
            aggregated_ps.append(p)

        aggregated_ps = np.array(aggregated_ps)

        reference_ps = []
        for _, e_sample in enumerate(reference_generator):
            p = get_p_matrix(e_sample)
            reference_ps.append(p)
        reference_ps = np.array(reference_ps)

        bar_agg = np.mean(aggregated_ps, axis=0)
        bar_ref = np.mean(reference_ps, axis=0)
        print(bar_agg)
        print(bar_ref)

        diff = np.abs(bar_ref - bar_agg)
        MAE = np.mean(diff)

        self.assertLessEqual(MAE, 0.025, f"MAE should be less than 0.025, was {MAE}")
        self.assertNotEqual(MAE, 0.00, f"MAE not be 0.0")




if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
