import tensorflow as tf
import numpy as np
from embedding_uncertainty.laplace_approx import subhessian, subhessian_analytic
from embedding_uncertainty.laplace_approx import full_hessian, fixed_inverse_hessian, laplace_approx
from embedding_uncertainty.laplace_approx import gradient_tf, gradient, hessian_tf
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.estimation import map_estimate
import unittest
import random, json, os

class Test(unittest.TestCase):
    def test_hessian(self):
        THRESHOLD = 0.00000001
        K = 3
        alpha = np.random.rand(K)
        rho = np.random.rand(K)

        n_plus = 3
        n_minus = 0
        H_truth = subhessian(n_plus, n_minus, rho, alpha)
        H_derived = subhessian_analytic(n_plus, n_minus, rho, alpha)

        MAE = tf.reduce_mean(tf.abs(H_truth - H_derived))
        self.assertLessEqual(MAE, THRESHOLD, f"Error between tf and direct calculation should be less than {THRESHOLD} ({MAE})")

        n_plus = 0
        n_minus = 2
        H_truth = subhessian(n_plus, n_minus, rho, alpha)
        H_derived = subhessian_analytic(n_plus, n_minus, rho, alpha)

        MAE = tf.reduce_mean(tf.abs(H_truth - H_derived))
        self.assertLessEqual(MAE, THRESHOLD, f"Error between tf and direct calculation should be less than {THRESHOLD} ({MAE})")

        n_plus = 5
        n_minus = 3
        H_truth = subhessian(n_plus, n_minus, rho, alpha)
        H_derived = subhessian_analytic(n_plus, n_minus, rho, alpha)

        MAE = tf.reduce_mean(tf.abs(H_truth - H_derived))
        self.assertLessEqual(MAE, THRESHOLD, f"Error between tf and direct calculation should be less than {THRESHOLD} ({MAE})")

    def test_hessian_different_ij(self):
        THRESHOLD = 0.00000001
        K = 3
        alpha = np.random.rand(K)
        rho = np.random.rand(K)

        n_plus = 5
        n_minus = 6
        H_truth = subhessian(n_plus, n_minus, rho, alpha, i=0, j=0)
        H_derived = subhessian_analytic(n_plus, n_minus, rho, alpha, i=0, j=0)

        MAE = tf.reduce_mean(tf.abs(H_truth - H_derived))
        self.assertLessEqual(MAE, THRESHOLD, f"Error between tf and direct calculation should be less than {THRESHOLD} ({MAE})")

        n_plus = 5
        n_minus = 6
        H_truth = subhessian(n_plus, n_minus, rho, alpha, i=1, j=1)
        H_derived = subhessian_analytic(n_plus, n_minus, rho, alpha, i=1, j=1)

        MAE = tf.reduce_mean(tf.abs(H_truth - H_derived))
        self.assertLessEqual(MAE, THRESHOLD, f"Error between tf and direct calculation should be less than {THRESHOLD} ({MAE})")


    def test_hessian_with_data(self):

        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data/1.json')
        MAP_FILENAME = os.path.join(os.path.dirname(__file__), 'data/map_estimate.json')

        N = 1000
        THRESHOLD = 0.00000001
        e_map = Embedding(saved_model_path=MAP_FILENAME)
        K = e_map.dimensionality
        V = len([wd for wd in e_map.vocabulary if "_c" not in wd])
        print(e_map.lambda0)
        e_map.lambda0 = e_map.dimensionality
        #print(e_map.lambda0)
        #print(e_map["word0"])
        #print(e_map["word1_c"])
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        data = []
        for elem in d["data"][:N]:
            w, v, x = elem["v"], elem["w"] + "_c", float(elem["x"])
            data.append((w,v,x))

        assert len(data) == N
        e_grad_tf = gradient_tf(e_map, data)
        e_grad = gradient(e_map, data)

        print(e_grad_tf["word0"])
        print(e_grad_tf["word2"])

        print(e_grad["word0"])
        print(e_grad["word2"])

        H_tf = hessian_tf(e_map, data, V, K)
        H = full_hessian(e_map, data)

        print(H_tf[:K, :K] - H[:K, :K])
        MAE = tf.reduce_mean(tf.abs(H_tf - H))
        self.assertLessEqual(MAE, THRESHOLD, f"Error between tf and direct calculation should be less than {THRESHOLD} ({MAE})")

    def test_laplace(self):

        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data/1.json')
        MAP_FILENAME = os.path.join(os.path.dirname(__file__), 'data/map_estimate.json')

        N = 100000
        THRESHOLD = 0.00000001
        e_map = Embedding(saved_model_path=MAP_FILENAME)
        K = e_map.dimensionality
        V = len([wd for wd in e_map.vocabulary if "_c" not in wd])
        print(e_map.lambda0)
        e_map.lambda0 = e_map.dimensionality
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        data = []
        for elem in d["data"][:N]:
            w, v, x = elem["v"], elem["w"] + "_c", float(elem["x"])
            data.append((w,v,x))

        assert len(data) == N
        e_grad_tf = gradient_tf(e_map, data)

        print(e_grad_tf["word0"])
        print(e_grad_tf["word2"])

        for sample in laplace_approx(e_map, data, samples=5, rotational_fix=True):
            print(sample["word0"])
            print(sample["word2"])