import numpy as np
from probabilistic_word_embeddings.embeddings import Embedding
import tensorflow as tf
from pathlib import Path
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
import progressbar
import pandas as pd

def get_matrix(e, words_w=None, words_c=None):
    if words_w is None:
        words_w = sorted([wd for wd in e.vocabulary if "_c" not in wd])
    if words_c is None:
        words_c = sorted([wd for wd in e.vocabulary if "_c" in wd])
        
    A = e[words_w]
    B = e[words_c]
    
    C = tf.tensordot(A, B, axes=[1,1])
    assert C.shape[0] == len(words_w)
    assert C.shape[1] == len(words_c)
    return C

def main(args):
    folder = Path(args.folder)
    runs = []
    bsruns = []
    for path in folder.glob(f"bs-{args.datasize}-dim-{args.dim}-ws-{args.ws}-no-*-bsno-*.pkl"):
        no = path.stem.split("-bsno-")[0].split("-no-")[-1]
        bsno = path.stem.split("-bsno-")[-1]
        runs.append(no)
        bsruns.append(bsno)
    runs = sorted(list(set(runs)))
    bsruns = sorted(list(set(bsruns)))
    
    R_batch = None
    estimated_means = None
    estimated_vars = None
    for run in runs:
        print("Run", run)
        C_batch = None
        for path in folder.glob(f"ref-{args.datasize}-dim-{args.dim}-ws-{args.ws}-no-{run}.pkl"):
            e = Embedding(saved_model_path=str(path.absolute()))
            R = get_matrix(e)
            if R_batch is None:
                R_batch = np.zeros([len(runs)] + R.shape)
            if estimated_vars is None:
                estimated_vars = np.zeros([len(runs)] + R.shape)
            if estimated_means is None:
                estimated_means = np.zeros([len(runs)] + R.shape)
            R_batch[int(run)] = R
        
        for path in progressbar.progressbar(folder.glob(f"bs-{args.datasize}-dim-{args.dim}-ws-{args.ws}-no-{run}-bsno-*.pkl")):
            no = int(path.stem.split("-bsno-")[0].split("-no-")[-1])
            bsno = int(path.stem.split("-bsno-")[-1])
            
            e = Embedding(saved_model_path=str(path.absolute()))
            C = get_matrix(e)
            
            if C_batch is None:
                C_batch = np.zeros([len(bsruns)] + C.shape)
            
            C_batch[bsno] = C
        print(C_batch.shape)
        estimated_means[int(run)] = tf.math.reduce_mean(C_batch, axis=0)
        estimated_vars[int(run)] = tf.math.reduce_std(C_batch, axis=0)

    print(R_batch.shape)
    
    true_mean = tf.math.reduce_mean(R_batch, axis=0)
    true_std =  tf.math.reduce_std(R_batch, axis=0)
    
    print(tf.reduce_mean(estimated_means, axis=0))
    print(true_mean)
    print(tf.reduce_mean(estimated_vars, axis=0))
    print(true_std)
    
    scatterplot = plot_bs_var_mean(true_std, tf.reduce_mean(estimated_vars, axis=0), samples=500)
    plt.savefig("bs_var_mean.png")
    plt.clf()
    scatterplot = plot_bs_var(true_std, estimated_vars, samples=500)
    plt.savefig("bs_var.png")
    plt.clf()
    scatterplot = histogram_bs_var_mean(true_std, tf.reduce_mean(estimated_vars, axis=0), samples=10000)
    plt.savefig("bs_var_mean_hist.png")
    
def histogram_bs_var_mean(true_std, estimated_std, samples=1000):
    i_shape, j_shape = true_std.shape
    print("true shape", true_std.shape)
    print("estimated_std shape", estimated_std.shape)
    rows = []
    for _ in progressbar.progressbar(range(samples)):
        i, j = random.randint(0, i_shape-1), random.randint(0, j_shape-1)
        true = true_std[i, j].numpy()
        estimated = estimated_std[i, j].numpy()
        ratio = estimated / true
        rows.append([ratio])
    df = pd.DataFrame(rows, columns=["ratio"])
    print(df)
    sns.set_theme()
    f, ax = plt.subplots(figsize=(7, 5))
    
    plot = sns.histplot(data=df, x="ratio", log_scale=True)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    return plot

def plot_bs_var_mean(true_std, estimated_std, samples=1000):
    i_shape, j_shape = true_std.shape
    print("true shape", true_std.shape)
    print("estimated_std shape", estimated_std.shape)
    rows = []
    for _ in progressbar.progressbar(range(samples)):
        i, j = random.randint(0, i_shape-1), random.randint(0, j_shape-1)
        true = true_std[i, j].numpy()
        estimated = estimated_std[i, j].numpy()
        rows.append([true, estimated])
    df = pd.DataFrame(rows, columns=["true", "estimated"])
    print(df)
    sns.set_theme()
    return sns.scatterplot(data=df, x="true", y="estimated")

def plot_bs_var(true_std, estimated_std, samples=1000):
    i_shape, j_shape = true_std.shape
    k_shape = estimated_std.shape[0]
    rows = []
    for _ in progressbar.progressbar(range(samples)):
        i, j, k = random.randint(0,i_shape-1), random.randint(0,j_shape-1), random.randint(0,k_shape-1)
        true = true_std[i, j].numpy()
        estimated = estimated_std[k, i, j]
        rows.append([true, estimated])
    df = pd.DataFrame(rows, columns=["true", "estimated"])
    print(df)
    sns.set_theme()
    return sns.scatterplot(data=df, x="true", y="estimated")

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--folder", type=str)
    argparser.add_argument("--dim", type=int, default=100)
    argparser.add_argument("--datasize", type=str, default="50M")
    argparser.add_argument("--ws", type=int, default=2)
    args = argparser.parse_args()
    print(args)
    main(args)
