from trainerlog import get_logger
LOGGER = get_logger("sampling", splitsec=True)
LOGGER.info("Load modules..")
import numpy as np
from polyagamma import random_polyagamma
import copy
import progressbar
import tensorflow as tf
import tqdm
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns

LOGGER.info("Done!")
from time import perf_counter as pc

def sample_directly_np(V_inv):
    V = np.linalg.inv(V_inv)
    return np.random.multivariate_normal(np.zeros(V_inv.shape[0]), V)

@tf.function
def sample_directly_tf(V_inv):
    K = V_inv.shape[0]
    V = tf.linalg.inv(V_inv)
    L = tf.linalg.cholesky(V)
    return tf.linalg.matvec(L, tf.random.normal([K], dtype=tf.float64))

@tf.function
def sample_via_solve(V_inv, COPIES=1):
    V_inv = tf.tile(tf.expand_dims(V_inv, axis=0), [COPIES,1,1])
    L_inv_T = tf.transpose(tf.linalg.cholesky(V_inv), perm=[0,2,1])
    y = tf.random.normal([COPIES, K, 1], dtype=tf.float64)
    #y = tf.linalg.matvec(V_inv, y)
    #print(V_inv.shape, y.shape)
    #V_inv = tf.tile(tf.expand_dims(V_inv, axis=0), [COPIES,1,1])


    #L_inv_T = tf.linalg.cholesky(V_inv)
    #y = tf.random.normal([K, 1], dtype=tf.float64)
    return tf.linalg.triangular_solve(L_inv_T, y, lower=False)

#@tf.function
def conjugate_gradient(A, b, tol=1e-10, max_iter=15):
    print(A.shape, b.shape)
    x = tf.zeros_like(b)
    r = b - tf.linalg.matvec(A, x)
    p = r
    rsold = tf.reduce_sum(r * r)
    
    for _ in range(max_iter):
        Ap = tf.linalg.matvec(A, p)
        alpha = rsold / tf.reduce_sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = tf.reduce_sum(r * r)
        #if tf.sqrt(rsnew) < tol:
        #    return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

@tf.function
def sample_via_indirect_solve(V_inv, COPIES = 50):
    #operator = tf.linalg.LinearOperatorFullMatrix(V_inv, is_self_adjoint=True, is_positive_definite=True)
    y = tf.random.normal([COPIES, K], dtype=tf.float64)
    #y = tf.linalg.matvec(V_inv, y)
    #print(V_inv.shape, y.shape)
    V_inv = tf.tile(tf.expand_dims(V_inv, axis=0), [COPIES,1,1])
    x = conjugate_gradient(V_inv, y)
    #print(result)
    #y = tf.random.normal([K, 1], dtype=tf.float64)
    return x
    #return tf.linalg.triangular_solve(L_inv_T, y, lower=False)

if __name__ == '__main__':
    K = 100

    duration_direct_tf, duration_via_solve, duration_via_id_solve = 0.0, 0.0, 0.0

    data = {"x1": [], "x2": [], "method": []}
    
    L_inv = np.random.randn(K, K) / np.sqrt(K)
    V_inv = L_inv.T @ L_inv


    for _ in tqdm.tqdm(range(300)):

        t0 = pc()
        x1 = sample_directly_np(V_inv)
        t1 = pc()
        data["x1"] += [x1[0]]
        data["x2"] += [x1[1]]
        data["method"] += ["numpy"]
        #print("time:", t1 - t0)

        V_inv = tf.constant(V_inv)
        t0 = pc()
        x2 = sample_directly_tf(V_inv)
        t1 = pc()
        data["x1"] += [x2[0]]
        data["x2"] += [x2[1]]
        data["method"] += ["naive"]
        duration_direct_tf += t1 - t0
        #print("time:", duration_direct_tf)

        V_inv = tf.constant(V_inv)
        COPIES = 50
        t0 = pc()
        x3 = sample_via_solve(V_inv, COPIES=COPIES)[0]
        t1 = pc()
        data["x1"] += [x3[0]]
        data["x2"] += [x3[1]]
        data["method"] += ["solve"]
        duration_via_solve += (t1 - t0)/COPIES

        V_inv = tf.constant(V_inv)
        COPIES = 200
        t0 = pc()
        x3 = sample_via_indirect_solve(V_inv, COPIES=COPIES)
        t1 = pc()
        #print(x3)
        x3 = x3[0]
        data["x1"] += [x3[0]]
        data["x2"] += [x3[1]]
        data["method"] += ["id-solve"]
        duration_via_id_solve += (t1 - t0) / COPIES
        #print("time:", duration_via_solve)

    df = pl.DataFrame(data)
    print(df)
    for method in sorted(list(set(df["method"]))):
        print(method)
        df_method = df.filter(pl.col("method") == method)
        print(df_method.select("x1", "x2").to_pandas().cov())
        print()


    print(tf.linalg.inv(V_inv)[:2, :2])
    #print(df).group_by("method").with_columns()
    print(f"{np.round(duration_direct_tf/ duration_via_solve, 2)}x improvement!")

    print(f"{np.round(duration_direct_tf/ duration_via_id_solve, 2)}x improvement!")

    sns.scatterplot(df, x="x1", y="x2", hue="method")
    plt.show()
