from embedding_uncertainty import polyagamma_sampler, polyagamma_sampler_tf
import numpy as np
import tensorflow as tf
from trainerlog import get_logger
LOGGER = get_logger("main")
LOGGER.info("Load modules..")
from time import sleep, perf_counter as pc
t0 = pc()
sleep(1)
print(pc()-t0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--M", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=128)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    ITERATIONS = args.iterations
    CHAINS = args.chains
    K, M = args.K, args.M
    beta = tf.random.normal([K]) / np.sqrt(K)
    X = tf.random.uniform([M, K])
    
    eta = tf.sigmoid(tf.linalg.matvec(X, beta))
    y = tf.cast(tf.random.uniform([len(eta)]) <= eta, dtype=tf.float32)
    N = tf.ones(M)
    beta0 = tf.zeros(K)
    
    t0 = pc()
    for sample in polyagamma_sampler(beta0.numpy(), X.numpy(), y.numpy(), iterations=ITERATIONS, N=N.numpy()):
        pass
    
    diff_singlethreaded = pc() - t0
    print("Timedelta singlethreaded", diff_singlethreaded, "(s)")
    print("CPU words per second", 1.0/(diff_singlethreaded))
    
    t1 = pc()
    beta0 = tf.stack([beta0 for _ in range(CHAINS)])
    X = tf.stack([X for _ in range(CHAINS)])
    y = tf.stack([y for _ in range(CHAINS)])
    N = tf.stack([N for _ in range(CHAINS)])
    print(X.shape)
    for sample in polyagamma_sampler_tf(beta0, X, y, iterations=ITERATIONS, N=N):
        pass
    
    diff_gpu = pc() - t1
    print("Timedelta GPU", diff_gpu / CHAINS, "(s)", f"(raw {diff_gpu})")
    print("GPU words per second", 1.0/(diff_gpu / CHAINS))