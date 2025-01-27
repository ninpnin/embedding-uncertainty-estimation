import tensorflow as tf
import numpy as np
from embedding_uncertainty.laplace_approx import subhessian, subhessian_analytic
from embedding_uncertainty.laplace_approx import full_hessian, fixed_inverse_hessian, laplace_approx
from embedding_uncertainty.laplace_approx import gradient_tf, gradient
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


    def test_laplace_approx(self):

        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data/1.json')
        MAP_FILENAME = os.path.join(os.path.dirname(__file__), 'data/map_estimate.pkl')

        THRESHOLD = 0.00000001
        e_map = Embedding(saved_model_path=MAP_FILENAME)
        print(e_map["word0"])
        print(e_map["word1_c"])
        with open(TESTDATA_FILENAME) as f:
            d = json.load(f)

        data = []
        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", float(elem["x"])
            data.append((w,v,x))

        e_grad = gradient_tf(e_map, data)

        print(e_grad["word0"])
        print(e_grad["word2"])

        for e_sample in laplace_approx(e_map, data, samples=10, rotational_fix=True):
            print(e_sample)
"""
    def test_full_hessian(self):
        THRESHOLD = 0.00000001
        K = 3
        V = 5
        lambda0 = 1.0
        vocab = {f"wd{v+1}" for v in range(V)}
        words = [wd for wd in vocab]
        contexts = [f"{wd}_c" for wd in vocab]
        e_ref = Embedding(vocab, dimensionality=K, lambda0=lambda0)

        for N in [1000, 5000, 30000]:
            BATCH_SIZE = N // 300

            data = []

            for _ in range(N):
                w = random.choice(words)
                v = random.choice(contexts)
                rho = e_ref[w]
                alpha = e_ref[v]
                nu = np.dot(alpha, rho) * 10
                eta = 1.0 / (1.0 + np.exp(-nu))
                x = 0.0
                if eta > random.uniform(0.0, 1.0):
                    x = 1.0

                data.append((w, v, x))

            def data_generator(batch_size):
                while True:
                    i, j, x = [], [], []
                    for _ in range(batch_size):
                        d = random.choice(data)
                        i.append(d["w"])
                        j.append(d["v"])
                        x.append(d["x"])

                    i = tf.constant(i)
                    j = tf.constant(j)
                    x = tf.constant(x, dtype=tf.float64)
                    yield (i, j, x)

            e = Embedding(vocab, dimensionality=K, lambda0=lambda0)
            e = map_estimate(e, data_generator=data_generator(BATCH_SIZE), N=N, batch_size=BATCH_SIZE, epochs=15)


            H = full_hessian(e, data)

            _, S, _ = np.linalg.svd(H, full_matrices=False)
            print("H singular vals", np.round(S, 3))

            Sigma = np.linalg.inv(H)
            _, S, _ = np.linalg.svd(Sigma, full_matrices=False)
            print("Sigma singular vals", np.round(S, 3))
            
            Sigma_aug = fixed_inverse_hessian(H, K)

            _, S, _ = np.linalg.svd(Sigma_aug, full_matrices=False)
            print("Sigma_aug singular vals", np.round(S, 3))
"""