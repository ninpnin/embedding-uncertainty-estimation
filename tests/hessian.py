import tensorflow as tf
import numpy as np
from embedding_uncertainty.laplace_approx import subhessian, subhessian_analytic
import unittest


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
