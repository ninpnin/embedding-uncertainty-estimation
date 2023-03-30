import numpy as np
import progressbar as pb
from pathlib import Path

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
    estim_probs = lm.get(x, dict())
    vocab = list(vocab)
    unnormalized_probs = {wd: estim_probs.get(wd, 0.0) + A for wd in vocab}
    valuesum = sum(unnormalized_probs.values())
    probs = {wd: val / valuesum for wd, val in unnormalized_probs.items()}
    theta = [probs[wd] for wd in vocab]
    x_prime = np.random.choice(len(vocab), p=theta)
    x_prime = vocab[x_prime]
    return x_prime


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/wikismall.txt")
    parser.add_argument("--outpath", type=str, default="data/markov/sample.txt")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=0.0001)
    args = parser.parse_args()
    
    text = open(args.data).read().lower().split()
    vocab = set(text)


    lm = get_probs(text)
    data = []

    x = np.random.choice(list(vocab))
    for i in range(args.N):
        data.append(x)
        x = sample(x, lm, vocab, A=args.smoothing)

    print(" ".join(data))

