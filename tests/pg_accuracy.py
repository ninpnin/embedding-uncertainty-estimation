import unittest

from embedding_uncertainty import polyagamma_sampler, polyagamma_sampler_tf
import numpy as np
import tensorflow as tf
import progressbar
import arviz as az
import math
class Test(unittest.TestCase):

    def test_pg_accuracy(self):
        """
        Test that the results of mean-field variational inference are coherent
        """
        WARMUP = 1
        ITERATIONS = WARMUP + 1
        CHAINS = 3500
        FULL_ITERATIONS = WARMUP + (ITERATIONS - WARMUP) * CHAINS
        print(FULL_ITERATIONS, "iterations in total.")
        K, M = 100, 2000
        beta = tf.random.normal([K]) / np.sqrt(K)
        X = tf.random.uniform([M, K])
        
        eta = tf.sigmoid(tf.linalg.matvec(X, beta))
        y = tf.cast(tf.random.uniform([len(eta)]) <= eta, dtype=tf.float32)
        N = tf.ones(M)
        beta0 = tf.zeros(K)

        samples_ground_truth = []
        samples = []
        for ix, sample in progressbar.progressbar(enumerate(polyagamma_sampler(beta0.numpy(), X.numpy(), y.numpy(), iterations=FULL_ITERATIONS, N=N.numpy()))):
            if ix >= WARMUP:
                samples_ground_truth.append(sample)

        for _ in progressbar.progressbar(range(CHAINS)):
            for ix, sample in enumerate(polyagamma_sampler(beta0.numpy(), X.numpy(), y.numpy(), iterations=ITERATIONS, N=N.numpy())):
                if ix >= WARMUP:
                    samples.append(sample)
                
        samples_ground_truth = np.array(samples_ground_truth)
        samples_mean_truth = np.mean(samples_ground_truth, axis=0)
        
        samples = np.array(samples)
        samples_mean = np.mean(samples, axis=0)
        self.assertEqual(samples_mean.shape, (K,))
        for k in range(K):
            ess1, ess2 = az.ess(samples[:,k]), az.ess(samples_ground_truth[:,k])
            MCSE1, MCSE2 = np.std(samples[:,k]) / np.sqrt(ess1), np.std(samples_ground_truth[:,k]) / np.sqrt(ess2)
            #print("ESS", ess1, "ESS", ess2)
            print("Dimension", k)
            decimals1, decimals2 = math.floor(-np.log10(MCSE1)), math.floor(-np.log10(MCSE2))
            #print("decimals1", decimals1, "decimals2", decimals2)
            decimals = min(decimals1, decimals2)
            
            print("Mean for sampling 50-60 iterations:", samples_mean[k], f"Mean for sampling {FULL_ITERATIONS} iterations:",samples_mean_truth[k], "true val", beta[k])
            print("MCSE1", MCSE1, "MCSE2", MCSE2)
            MCSE_total = np.sqrt((MCSE1 ** 2) + (MCSE2 ** 2))
            print("Mean diff", samples_mean[k] - samples_mean_truth[k], "MCSE_total", MCSE_total)
            
            samples_k = np.array([samples[:,k], samples_ground_truth[:,k]])
            rhat_k = az.rhat(samples_k)
            print("Rhat between short and long chains", rhat_k)
            self.assertLessEqual(rhat_k, 1.01)

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
