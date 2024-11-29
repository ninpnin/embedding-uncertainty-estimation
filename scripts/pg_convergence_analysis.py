from embedding_uncertainty import polyagamma_sampler, polyagamma_sampler_tf
import numpy as np
import tensorflow as tf
import progressbar
import arviz as az
import math
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
#from trainerlog import get_logger

#get_logger("")
dtype = tf.float64

def get_random_beta(K, epsilon_signal):
    return epsilon_signal * tf.random.normal([K], dtype=dtype) / np.sqrt(K)

def get_likelihoods(betas, X, y):
    nu = tf.linalg.matvec(X, betas) * y
    eta = tf.sigmoid(nu)
    log_eta = tf.math.log(eta)
    #print(log_eta.shape)
    return tf.reduce_sum(tf.math.log(eta), axis=-1)

def main():
    K, M = 50, 150

    epsilon_signal = 2.0
    beta = get_random_beta(K, epsilon_signal)
    X = tf.random.uniform([M, K], dtype=dtype)
    
    eta = tf.sigmoid(tf.linalg.matvec(X, beta))
    y = tf.cast(tf.random.uniform([len(eta)], dtype=dtype) <= eta, dtype=dtype)
    N = tf.ones(M, dtype=dtype)

    CHAINS = 4000
    beta0 = tf.stack([get_random_beta(K, epsilon_signal) for _ in range(CHAINS)])
    X = tf.stack([X for _ in range(CHAINS)])
    y = tf.stack([y for _ in range(CHAINS)])
    N = tf.stack([N for _ in range(CHAINS)])
    X = tf.cast(X, dtype=tf.float64)
    y = tf.cast(y, dtype=tf.float64)
    N = tf.cast(N, dtype=tf.float64)
    beta0 = tf.cast(beta0, dtype=tf.float64)
    
    column_names = [f"beta{k}" for k in range(K)]
    dfs = []

    warmups = [0, 1, 10, 50]
    for WARMUP in warmups:
        ITERATIONS = WARMUP + 1

        samples_ground_truth = []
        for ix, sample in progressbar.progressbar(enumerate(polyagamma_sampler_tf(beta0, X, y, iterations=ITERATIONS, N=N, multivariate_method="cholesky"))):
            if ix >= WARMUP:
                samples_ground_truth.append(sample)

        samples_ground_truth = np.array(samples_ground_truth)[0]

        ll = get_likelihoods(samples_ground_truth, X[0], y[0]).numpy()
        print("ll.shape", ll.shape)


        df = pl.from_numpy(samples_ground_truth, schema=column_names)
        df = df.with_columns(pl.lit(f"{WARMUP}").alias("warmup"))
        df = df.with_columns(pl.Series("log_likelihood", ll))
        dfs.append(df)
        #exit()
    
    df = pl.concat(dfs)
    print(df)
        
    for col in ["log_likelihood"] + column_names:
        wu_last = warmups[-1]
        df_truth = df.filter(pl.col("warmup") == f"{wu_last}")
        samples_k_truth = np.array(df_truth[col])
        for wu in warmups[:-1]:
            df_wu = df.filter(pl.col("warmup") == f"{wu}")
            samples_k_wu = np.array(df_wu[col])
            samples_k = np.array([samples_k_truth, samples_k_wu])
            
            #print("Samples_k shape", samples_k.shape)
            rhat_k = az.rhat(samples_k)
            print(f"Rhat with warmup = {wu}", rhat_k)
        

        sns.kdeplot(df, x=col, hue="warmup")
        plt.show()

if __name__ == '__main__':
    # begin the unittest.main()
    main()
