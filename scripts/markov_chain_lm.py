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

def sample(x, lm, vocab, A=0.5):
    probs = lm.get(x, dict())
    theta = [probs[wd] for wd in vocab]
    x_prime = np.random.choice(len(vocab), p=theta)
    x_prime = vocab[x_prime]
    return x_prime

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

    data = []

    x = np.random.choice(list(vocab))
    for i in progressbar.progressbar(range(args.N)):
        data.append(x)
        x = sample(x, lm, vocab, A=args.smoothing)

    print(" ".join(data))

