import unittest

from embedding_uncertainty import polyagamma_sampler, polyagamma_sampler_tf
import numpy as np
import tensorflow as tf
import progressbar
import arviz as az
import math

WARMUP = 1
ITERATIONS = WARMUP + 1
CHAINS = 2000
FULL_ITERATIONS = WARMUP + (ITERATIONS - WARMUP) * CHAINS

class Test(unittest.TestCase):


    def test_pg_accuracy(self):
        """
        Test that the results of mean-field variational inference are coherent
        """
        for K, M in []: # [(100, 500)]
            print(FULL_ITERATIONS, "iterations in total.")
            #K, M = 100, 500
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
                print("Samples_k shape", samples_k.shape)
                rhat_k = az.rhat(samples_k)
                print("Rhat between short and long chains", rhat_k)
                self.assertLessEqual(rhat_k, 1.01)

    def test_pg_gpu_accuracy(self):
        """
        Check that CPU and GPU samplers yield the same result
        """
        dtype = tf.float64
        for hyperparams in [(5, 200), (5, 20), (20, 15), (20, 150), (100, 70), (100, 300)]:
            K, M = hyperparams
            beta = tf.random.normal([K], dtype=dtype) / np.sqrt(K)
            X = tf.random.uniform([M, K], dtype=dtype)
            
            eta = tf.sigmoid(tf.linalg.matvec(X, beta))
            y = tf.cast(tf.random.uniform([len(eta)], dtype=dtype) <= eta, dtype=dtype)
            N = tf.ones(M, dtype=dtype)
            beta0 = tf.zeros(K, dtype=dtype)

            samples_ground_truth = []
            for ix, sample in progressbar.progressbar(enumerate(polyagamma_sampler(beta0.numpy(), X.numpy(), y.numpy(), iterations=FULL_ITERATIONS, N=N.numpy()))):
                if ix >= WARMUP:
                    samples_ground_truth.append(sample)

            samples_ground_truth = np.array(samples_ground_truth)
            
            beta0 = tf.stack([beta0 for _ in range(CHAINS)])
            X = tf.stack([X for _ in range(CHAINS)])
            y = tf.stack([y for _ in range(CHAINS)])
            N = tf.stack([N for _ in range(CHAINS)])
            X = tf.cast(X, dtype=tf.float64)
            y = tf.cast(y, dtype=tf.float64)
            N = tf.cast(N, dtype=tf.float64)
            beta0 = tf.cast(beta0, dtype=tf.float64)
            
            samples_gpu = []
            for ix, sample in progressbar.progressbar(enumerate(polyagamma_sampler_tf(beta0, X, y, iterations=ITERATIONS, N=N))):
                if ix >= WARMUP:
                    samples_gpu.append(sample)
            
            samples_gpu = np.array(samples_gpu)[0]
            print("GPU", samples_gpu.shape, "CPU", samples_ground_truth.shape)
            samples_mean_truth = np.mean(samples_ground_truth, axis=0)
            samples_mean = np.mean(samples_gpu, axis=0)

            samples_std_truth = np.std(samples_ground_truth, axis=0)
            samples_std = np.std(samples_gpu, axis=0)

            samples_max_truth = np.max(samples_ground_truth, axis=0)
            samples_max = np.max(samples_gpu, axis=0)
            samples_argmax = np.argmax(samples_gpu, axis=0)
            print("samples argmax",samples_argmax )
            
            
            for k in range(K):
                print("Mean for GPU:", samples_mean[k], f"Mean CPU:", samples_mean_truth[k], "true val", beta[k])
                print("Stdev for GPU:", samples_std[k], f"Stdev CPU:", samples_std_truth[k], "true val", beta[k])
                print("Max for GPU:", samples_max[k], f"Max CPU:", samples_max_truth[k], "true val", beta[k])
                samples_k = np.array([samples_gpu[:,k], samples_ground_truth[:,k]])
                print("Samples_k shape", samples_k.shape)
                rhat_k = az.rhat(samples_k)
                print("Rhat between short and long chains", rhat_k)
                self.assertLessEqual(rhat_k, 1.05, f"Rhat between short and long chains too high {rhat_k} for dim {k} of K={K} and N={M} and {ITERATIONS} samples")

    def test_pg_gpu_priorinit_accuracy(self):
        """
        Check that CPU and GPU samplers yield the same result
        when GPU sampler is initialized from the prior
        """
        dtype = tf.float64
        for hyperparams in [(5, 200), (5, 20), (20, 15), (20, 150), (100, 70), (100, 300)]:
            K, M = hyperparams
            beta = tf.random.normal([K], dtype=dtype) / np.sqrt(K)
            X = tf.random.uniform([M, K], dtype=dtype)
            
            eta = tf.sigmoid(tf.linalg.matvec(X, beta))
            y = tf.cast(tf.random.uniform([len(eta)], dtype=dtype) <= eta, dtype=dtype)
            N = tf.ones(M, dtype=dtype)
            beta0 = tf.zeros(K, dtype=dtype)

            samples_ground_truth = []
            for ix, sample in progressbar.progressbar(enumerate(polyagamma_sampler(beta0.numpy(), X.numpy(), y.numpy(), iterations=FULL_ITERATIONS, N=N.numpy()))):
                if ix >= WARMUP:
                    samples_ground_truth.append(sample)

            samples_ground_truth = np.array(samples_ground_truth)
            
            # Initialize betas from the prior
            beta0 = tf.stack([tf.random.normal([K], dtype=dtype) / np.sqrt(K) for _ in range(CHAINS)])
            X = tf.stack([X for _ in range(CHAINS)])
            y = tf.stack([y for _ in range(CHAINS)])
            N = tf.stack([N for _ in range(CHAINS)])
            X = tf.cast(X, dtype=tf.float64)
            y = tf.cast(y, dtype=tf.float64)
            N = tf.cast(N, dtype=tf.float64)
            beta0 = tf.cast(beta0, dtype=tf.float64)
            
            samples_gpu = []
            for ix, sample in progressbar.progressbar(enumerate(polyagamma_sampler_tf(beta0, X, y, iterations=ITERATIONS, N=N))):
                if ix >= WARMUP:
                    samples_gpu.append(sample)
            
            samples_gpu = np.array(samples_gpu)[0]
            print("GPU", samples_gpu.shape, "CPU", samples_ground_truth.shape)
            samples_mean_truth = np.mean(samples_ground_truth, axis=0)
            samples_mean = np.mean(samples_gpu, axis=0)

            samples_std_truth = np.std(samples_ground_truth, axis=0)
            samples_std = np.std(samples_gpu, axis=0)


            samples_max_truth = np.max(samples_ground_truth, axis=0)
            samples_max = np.max(samples_gpu, axis=0)
            samples_argmax = np.argmax(samples_gpu, axis=0)
            print("samples argmax",samples_argmax )
            
            
            for k in range(K):
                print("Mean for GPU:", samples_mean[k], f"Mean CPU:", samples_mean_truth[k], "true val", beta[k])
                print("Stdev for GPU:", samples_std[k], f"Stdev CPU:", samples_std_truth[k], "true val", beta[k])
                print("Max for GPU:", samples_max[k], f"Max CPU:", samples_max_truth[k], "true val", beta[k])
                samples_k = np.array([samples_gpu[:,k], samples_ground_truth[:,k]])
                print("Samples_k shape", samples_k.shape)
                rhat_k = az.rhat(samples_k)
                print("Rhat between short and long chains", rhat_k)
                self.assertLessEqual(rhat_k, 1.05, f"Rhat between short and long chains too high {rhat_k} for dim {k} of K={K} and N={M} and {ITERATIONS} samples")

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
