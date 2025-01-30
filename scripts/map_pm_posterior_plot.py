from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.models import sgns_likelihood
from probabilistic_word_embeddings.evaluation import posterior_mean, nearest_neighbors
from probabilistic_word_embeddings.utils import align
import numpy as np
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
from pathlib import Path
import random, json
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import tqdm
import arviz as az

def cossim(a,b):
    return np.dot(a, b) /(np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_folder", type=str, default=None, nargs="+")
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--words", type=str, nargs="+", default=["word0"])
    parser.add_argument("--plot_type", type=str, default="cossim")
    parser.add_argument("--align_samples", type=bool, default=False)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    
    sample_folders = [Path(sample_folder) for sample_folder in args.sample_folder]
    ref_emb = Embedding(saved_model_path=str(list(sample_folders[0].glob("*.pkl"))[-1].absolute()))
    dimensionality = ref_emb.dimensionality
    
    e_map = Embedding(saved_model_path=args.map_path)
    
    # Discard warmup samples
    chains = []
    for sample_folder in sample_folders:
        samples = sorted(sample_folder.glob("*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
        # By default the first half
        if args.warmup is None:
            samples = samples[len(samples) // 2:]

        chains.append(samples)
        
    #e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    
    rows = []
    for chain_ix, samples in enumerate(chains):
        for ix, sample in tqdm.tqdm(enumerate(samples)):
            e_sample = Embedding(saved_model_path=str(sample.absolute()))
            if args.align_samples:
                e_sample = align(e_sample, e_map, [wd for wd in e_sample.vocabulary])
            values = e_sample[args.words][0, :2].numpy().tolist()
            alpha = e_sample["word0_c"][:2].numpy().tolist()
            similarity = cossim(e_sample["word0"], e_sample["word1"])
            norm = np.linalg.norm(e_sample["word0"])
            rows.append([chain_ix, ix, similarity, norm] + values + alpha)

    df = pd.DataFrame(rows, columns=["chain", "ix", "cossim", "norm", "rho11", "rho12", "alpha00", "alpha01"])
    print(df)
    print(df.mean())

    print("rho", e_map[args.words][0, :2])
    print("alpha", e_map["word0_c"][ :2])

    CHAINS = len(chains)
    for chain in range(CHAINS):
        rho11_chain = np.array(df[df["chain"] == chain]["rho11"])
        ESS_rho11_chain0 = az.ess(rho11_chain)
        print(f"ESS chain  {chain} rho11", ESS_rho11_chain0)
        print(f"MCSE for chain {chain} rho11", np.std(rho11_chain) /np.sqrt(ESS_rho11_chain0))

        cossim_chain = np.array(df[df["chain"] == chain]["cossim"])
        ESS_cossim_chain0 = az.ess(cossim_chain)
        print(f"ESS chain  {chain} cossim", ESS_cossim_chain0)
        print(f"MCSE for chain {chain} cossim", np.std(cossim_chain) /np.sqrt(ESS_cossim_chain0))

    if args.plot_type == "cossim":
        sns.kdeplot(data=df, x="cossim", hue="chain", common_norm=False)
        print("Cossim MAP", cossim(e_map["word0"], e_map["word1"]))
    elif args.plot_type == "norm":
        sns.kdeplot(data=df, x="norm", hue="chain", common_norm=False)
        print("Cossim MAP", np.linalg.norm(e_map["word0"]) )
    elif args.plot_type == "kde":
        #print(e_post_mean[args.words][0, :2])
        #sns.scatterplot(df, x="rho11", y="rho12")
        sns.kdeplot(data=df, x="rho11", y="rho12", hue="chain", common_norm=False)
        #sns.kdeplot(data=df, x="rho11", y="rho12")
        #sns.scatterplot(data=df[df["chain"] == 0], x="rho11", y="rho12", hue="ix")
        plt.scatter(e_map[args.words][0, 0], e_map[args.words][0, 1], color='red', marker='x', s=100, linewidth=2)
    plt.show()

    print("Corr", df[["rho11", "rho12"]].corr())

    rho11_chain2 = np.array(df[df["chain"] == CHAINS-1]["cossim"])
    rho11_chain1 = np.array(df[df["chain"] == CHAINS-2]["cossim"])

    L = min(len(rho11_chain2), len(rho11_chain2))

    rho11_chain2 = rho11_chain2[:L]
    rho11_chain1 = rho11_chain1[:L]
    rho11_chains = np.stack([rho11_chain1, rho11_chain2])
    RHat = az.rhat(rho11_chains)
    print("Rhat cossim", RHat)

    rho11_chain2 = np.array(df[df["chain"] == CHAINS-1]["rho11"])
    rho11_chain1 = np.array(df[df["chain"] == CHAINS-2]["rho11"])
    rho11_chain2 = rho11_chain2[:L]
    rho11_chain1 = rho11_chain1[:L]
    rho11_chains = np.stack([rho11_chain1, rho11_chain2])
    RHat = az.rhat(rho11_chains)
    print("Rhat rho11", RHat)

    rho11_chain2 = np.array(df[df["chain"] == CHAINS-1]["rho12"])
    rho11_chain1 = np.array(df[df["chain"] == CHAINS-2]["rho12"])
    rho11_chain2 = rho11_chain2[:L]
    rho11_chain1 = rho11_chain1[:L]
    rho11_chains = np.stack([rho11_chain1, rho11_chain2])
    RHat = az.rhat(rho11_chains)
    print("Rhat rho12", RHat)