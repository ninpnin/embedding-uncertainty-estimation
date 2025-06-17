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
import re

COLORS = {"HMC": '#1f77b4', "MFVI": '#ff7f0e', "Gibbs": '#2ca02c', "Laplace": "#df647a"}
def increment_number_in_string(s):
    # Extract the number from the string
    number = re.search(r'\d+', s)
    if number:
        # Convert to integer, increment, and convert back to string
        incremented = str(int(number.group()) + 1)
        # Replace the original number with the incremented one
        s = re.sub(r'\d+', incremented, s, count=1)
    s = s.replace("word", "")
    return s

def cossim(a,b):
    return np.dot(a, b) /(np.linalg.norm(a) * np.linalg.norm(b))

def r_hat(df, reference="HMC"):
    import arviz as az
    methods = list(set(df["Method"]))
    print(methods)
    
    hmc = np.array(df[df["Method"] == reference]["similarity"])
    print(hmc)
    
    n = len(hmc) // 2
    x_start = hmc[:n]
    x_end = hmc[-n:]
    X = np.array([x_start, x_end])
    R = az.rhat(X)
    ESS = az.ess(hmc)
    print("Chain ESS:", ESS)
    
    print(f"Within chain R hat ({reference}):", R)
        
    other_methods = [m for m in methods if m != reference]
    for om in other_methods:
        print("Compare", reference, "to", om)
        x = np.array(df[df["Method"] == om]["similarity"])
        minlen = min(len(x), len(hmc))
        print("Shorter chain length:", minlen)
        print("Truncate both chains to that")
        x = x[:minlen]
        y = hmc[:minlen]
        ESS = az.ess(x)
        print("Chain ESS:", ESS)
        
        X = np.array([x, y])
        assert X.shape[0] < X.shape[1], "Chains go first in the ndarray"
        R = az.rhat(X)
        print("Between chain R hat:", R)
        
        n = len(x) // 2
        x_start = x[:n]
        x_end = x[-n:]
        
        X = np.array([x_start, x_end])
        R = az.rhat(X)
        print("Within chain R hat:", R)
        
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_folder", type=str, default=None, nargs="+")
    parser.add_argument("--warmup", type=int, default=None, nargs="+")
    parser.add_argument("--cossim_words", type=str, default=[], nargs="+")
    parser.add_argument("--do_wordsim", type=bool, default=False)
    parser.add_argument("--full_xaxis", type=bool, default=False, help="Force the x axis to range from -1 to 1")
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")
    
    # Discard warmup samples
    if args.warmup is None:
        args.warmup = [None] * len(args.sample_folder)
    elif len(args.warmup) == 1:
        args.warmup = [args.warmup] * len(args.sample_folder)
    else:
        assert len(args.warmup) == len(args.sample_folder)
        
    print("warmup", args.warmup)
    chains = []
    for sample_folder, warmup in zip(args.sample_folder, args.warmup):
        sample_folder = Path(sample_folder)
        samples = sorted(sample_folder.glob("*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
        # By default the first half
        if warmup is None:
            samples = samples[len(samples) // 2:]
        else:
            samples = samples[warmup:]
        chains.append(samples)
        
    #e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    
    rows = []
    cossim_rows = []
    for chain_ix, samples in enumerate(chains):
        for ix, sample in tqdm.tqdm(list(enumerate(samples))):
            LOGGER.debug(f"Load data from {str(sample.absolute())}...")
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
        
        # Hard code methods to certain colors        
        method_order = [methods[chain_ix] for chain_ix in range(max(chain_info)+1)]
        custom_palette = sns.color_palette([COLORS[method] for method in method_order])
        sns.set_palette(custom_palette)

        g = sns.kdeplot(cossim_results, x="similarity", hue="Method", common_norm=False, linewidth=2.0, legend=False)
        sns.despine()
        plt.ylabel(None, fontsize=24)
        w1, w2 = [increment_number_in_string(s) for s in args.cossim_words]

        plt.xlabel(f"cossim(ρ_{w1}, ρ_{w2})", fontsize=13)
        #plt.yticks(fontsize=0)
        #plt.yticks([], [])
        #ax = plt.gca()
        #ax.spines['left'].set_visible(False)
        #ax.get_yaxis().set_visible(False)
        TICKSIZE = 15
        plt.xticks(fontsize=TICKSIZE)
        plt.yticks(fontsize=TICKSIZE)
        #plt.legend(fontsize=20)
        #ax = plt.gca()
        if args.full_xaxis:
            plt.xlim(-1.2, 1.2)
            plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

        plt.savefig(f"img/{args.cossim_words[0]}-{args.cossim_words[1]}-similarity.pdf")
        plt.show()
        
        r_hat(cossim_results, reference="Gibbs")

    results = pd.concat(rows)
    print(results)

    print(results.groupby("Dataset").mean(numeric_only=True))
    print(results.groupby("Dataset").quantile(0.05, numeric_only=True))
    print(results.groupby("Dataset").quantile(0.95, numeric_only=True))

    print(results.groupby("ix").mean(numeric_only=True).quantile(0.05, numeric_only=True))
    print(results.groupby("ix").mean(numeric_only=True).quantile(0.95, numeric_only=True))


