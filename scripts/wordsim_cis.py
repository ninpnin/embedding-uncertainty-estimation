from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.models import sgns_likelihood
from probabilistic_word_embeddings.evaluation import posterior_mean, nearest_neighbors
from probabilistic_word_embeddings.evaluation import evaluate_word_similarity
import numpy as np
import tensorflow as tf
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
from pathlib import Path
import random, json
import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def cossim(a,b):
    return np.dot(a, b) /(np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_folder", type=str, default=None, nargs="+")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cossim_words", type=str, default=[], nargs="+")
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    
    # Discard warmup samples
    chains = []
    for sample_folder in args.sample_folder:
        sample_folder = Path(sample_folder)
        samples = sorted(sample_folder.glob("*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
        # By default the first half
        if args.warmup is None:
            samples = samples[len(samples) // 2:]
        chains.append(samples)
        
    #e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    
    rows = []
    cossim_rows = []
    for chain_ix, samples in enumerate(chains):
        for ix, sample in tqdm.tqdm(list(enumerate(samples))):
            e_sample = Embedding(saved_model_path=str(sample.absolute()))
            results = evaluate_word_similarity(e_sample)
            #print(results)
            results["chain"] = chain_ix
            results["ix"] = ix

            if len(args.cossim_words) == 2:
                word1, word2 = args.cossim_words
                similarity = cossim(e_sample[word1], e_sample[word2])
                cossim_rows.append([chain_ix, ix, similarity])

            rows.append(results)

    if len(args.cossim_words) == 2:
        cossim_results = pd.DataFrame(cossim_rows, columns=["chain", "ix", "similarity"])
        print(cossim_results)
        method1, method2 = ["Gibbs" if "vi" not in folder.lower() else "MFVI" for folder in args.sample_folder[:2]]
        chain_info = list(cossim_results["chain"])
        cossim_results["Method"] = [method1 if chain_ix == 0 else method2 for chain_ix in chain_info]
        sns.kdeplot(cossim_results, x="similarity", hue="Method", common_norm=False)
        sns.despine()
        plt.savefig(f"{args.cossim_words[0]}-{args.cossim_words[1]}-similarity.pdf")
        plt.show()

    results = pd.concat(rows)
    print(results)

    print(results.groupby("Dataset").mean(numeric_only=True))
    print(results.groupby("Dataset").quantile(0.05, numeric_only=True))
    print(results.groupby("Dataset").quantile(0.95, numeric_only=True))

    print(results.groupby("ix").mean(numeric_only=True).quantile(0.05, numeric_only=True))
    print(results.groupby("ix").mean(numeric_only=True).quantile(0.95, numeric_only=True))


