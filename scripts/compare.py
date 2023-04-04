import numpy as np
from pathlib import Path
from numpy import dot
from numpy.linalg import norm
from probabilistic_word_embeddings.embeddings import Embedding
from copy import deepcopy

def cossim(a,b):
    return dot(a, b) / (norm(a) * norm(b))

def get_words(folder):
    files = folder.glob("rho_*")
    return [f.stem.split("rho_")[-1] for f in files]

def hmc_intervals(target, words, samples=500):
    print("HMC (conditional)")
    print()

    embs = {}
    ranges = {}

    for wd in words:
        print(wd)
        embs[wd] = np.load(f"trained/rho_{wd}.npy")

    e_target = embs[target]
    N, D = e_target.shape
    print(N, D)
    for wd in words:
        similarities = []
        if wd == target:
            continue

        e_compar = embs[wd]

        i = np.random.randint(N, size=samples)
        j = np.random.randint(N, size=samples)
        
        for n in range(samples):
            rho_in = e_target[i[n]]
            rho_jn = e_compar[j[n]]

            sim = cossim(rho_in, rho_jn)
            similarities.append(sim)

        print(wd, np.mean(similarities), np.std(similarities)) 
        ranges[wd] = np.array(similarities)

def vi_intervals(target, words, samples=250):
    print("Variational Inference")
    print()
    ranges = {}

    e_mean = Embedding(saved_model_path="vi_mean.pkl")
    e_std = deepcopy(e_mean)
    e_std_val = np.load("vi_std.pkl.npy")
    e_std.theta.assign(e_std_val)

    for wd in words:
        similarities = []
        if wd == target:
            continue
        
        for n in range(samples):
            rho_in = e_mean[target] + np.random.randn(100) * e_std[target]
            rho_jn = e_mean[wd] + np.random.randn(100) * e_std[wd]

            sim = cossim(rho_in, rho_jn)
            similarities.append(sim)

        print(wd, np.mean(similarities), np.std(similarities)) 
        ranges[wd] = np.array(similarities)


def bootstrap_intervals(target, words):
    print("Bootstrap")
    print()

    ranges = {}
    for path in Path("trained/").glob("boostrap*.pkl"):
        e = Embedding(saved_model_path=str(path.absolute()))
        rho_target = e[target]
        for wd in words:
            rho_wd = e[wd]
            sim = cossim(rho_target, rho_wd)
            ranges[wd] = ranges.get(wd, []) + [sim]

    for wd, r in ranges.items():
        print(wd, np.mean(r), np.std(r)) 
        
    return ranges

if __name__ == "__main__":
    folder = Path("trained")
    target = "dog"
    wds = get_words(folder)
    print(wds)
    hmc = hmc_intervals(target, wds)
    boostrap = bootstrap_intervals(target, wds)
    vi = vi_intervals(target, wds)
