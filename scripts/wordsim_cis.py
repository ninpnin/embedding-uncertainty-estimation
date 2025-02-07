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
    parser.add_argument("--do_wordsim", type=bool, default=False)
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
        else:
            samples = samples[args.warmup:]
        chains.append(samples)
        
    #e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    
    rows = []
    cossim_rows = []
    for chain_ix, samples in enumerate(chains):
        for ix, sample in tqdm.tqdm(list(enumerate(samples))):
            e_sample = Embedding(saved_model_path=str(sample.absolute()))
            results = None
            if args.do_wordsim:
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
        def estimator_name(foldername):
            if "vi" in foldername.lower():
                return "MFVI"
            elif "laplace" in foldername.lower():
                return "Laplace"
            elif "hmc" in foldername.lower():
                return "HMC"
            else:
                return "Gibbs"
        cossim_results = pd.DataFrame(cossim_rows, columns=["chain", "ix", "similarity"])
        print(cossim_results)
        methods = [estimator_name(folder) for folder in args.sample_folder]
        chain_info = list(cossim_results["chain"])
        cossim_results["Method"] = [methods[chain_ix] for chain_ix in chain_info]
        g = sns.kdeplot(cossim_results, x="similarity", hue="Method", common_norm=False, legend=False, linewidth=3)
        sns.despine()
        #g.legend(fontsize=20)
        #plt.figure(figsize=(10, 6))
        sns.set_theme(rc={'figure.figsize':(10, 6)})
        plt.xlabel(None, fontsize=24)
        #plt.yticks(fontsize=0)
        #plt.yticks([], [])
        ax = plt.gca()
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.xticks(fontsize=24)
        #plt.legend(fontsize=20)
        #ax = plt.gca()
        plt.xlim(-1.2, 1.2)
        plt.rcParams["font.family"] = "cursive"


        plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        plt.savefig(f"img/{args.cossim_words[0]}-{args.cossim_words[1]}-similarity.pdf")
        plt.show()

    results = pd.concat(rows)
    print(results)

    print(results.groupby("Dataset").mean(numeric_only=True))
    print(results.groupby("Dataset").quantile(0.05, numeric_only=True))
    print(results.groupby("Dataset").quantile(0.95, numeric_only=True))

    print(results.groupby("ix").mean(numeric_only=True).quantile(0.05, numeric_only=True))
    print(results.groupby("ix").mean(numeric_only=True).quantile(0.95, numeric_only=True))


