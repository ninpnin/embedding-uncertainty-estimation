import numpy as np
import progressbar as pb
from pathlib import Path
import progressbar

def get_probs(text):
    lm = {}
    last_word, first_word = text[-1], text[0]
    lm[last_word] = {first_word: 1.0}
    for wd0, wd1 in zip(text[:-1], text[1:]):
        lm_wd0 = lm.get(wd0, dict())
        lm_wd0[wd1] = lm_wd0.get(wd1, 0.0) + 1.0
        lm[wd0] = lm_wd0

    return lm

def smoothen_lm(lm, A=0.5):
    C = len(lm)
    #print(f"Smoothen lm with constant {A}")
    A = A / C
    #print(f"Adjusted smoothing constant {A}")
    for wd in lm:
        unnormalized_probs = lm[wd]
        unnormalized_probs = {wd: unnormalized_probs.get(wd, 0.0) + A for wd in lm}
        valuesum = sum(unnormalized_probs.values())
        probs = {wd: val / valuesum for wd, val in unnormalized_probs.items()}
        lm[wd] = probs

    return lm

def get_pmi_matrix(lm, ws=1):
    #print(lm)
    vocab = [wd for wd in lm]
    A = np.zeros((len(vocab), len(vocab)))
    for i, w1 in enumerate(vocab):
        for j, w2 in enumerate(vocab):
            p = lm[w1][w2]
            A[j,i] = p
    
    A = A.T
    print(A)
    A_lim = A
    for _ in range(10000):
        A_lim = A_lim @ A
    
    print("LIM:")
    print(A_lim[0])
    
    v_max = A_lim[0]
    v_max = v_max / np.sum(v_max)
    print(v_max)
    print("Most freq:", vocab[list(v_max).index(max(v_max))])
    B = np.zeros((len(vocab), len(vocab)))
    A_prime = np.identity(len(vocab))
    for w in range(ws):
        A_prime = A_prime @ A
        
        B += 0.5/ws * A_prime
        B += 0.5/ws * A_prime.T
    
    #print(B)
    PMI = B / np.outer(v_max, v_max)
    PMI = np.real(PMI)
    log_PMI = np.log(PMI)
    return log_PMI, vocab

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/wikismall.txt")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=0.0001)
    args = parser.parse_args()
    
    text = open(args.data).read().lower()
    text = text.replace(",", "").replace(".", "").replace("?", "")
    text = text.split()
    #print(text[:10])
    vocab = list(set(text))

    lm = get_probs(text)
    lm = smoothen_lm(lm, A=args.smoothing)

    log_PMI, vocab = get_pmi_matrix(lm)
    print(vocab)
    print(log_PMI)
