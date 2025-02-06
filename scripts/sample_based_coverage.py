from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.models import sgns_likelihood
from probabilistic_word_embeddings.evaluation import posterior_mean, nearest_neighbors
from probabilistic_word_embeddings.evaluation import evaluate_word_similarity
import numpy as np
import tensorflow as tf
import tqdm
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
from pathlib import Path
import random, json
import polars as pl

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sample_folder", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--ci_level", type=float, default=0.9)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    sample_folder = Path(args.sample_folder)
    ref_emb = Embedding(saved_model_path=str(list(sample_folder.glob("*.pkl"))[-1].absolute()))
    K_sample, V_sample = ref_emb.dimensionality, len(ref_emb.vocabulary) // 2
    LOGGER.train(f"Sample K: {ref_emb.dimensionality}, V: { len(ref_emb.vocabulary) // 2}")
    
    

    # Discard warmup samples
    samples = sorted(sample_folder.glob("*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
    # By default the first half
    if args.warmup is None:
        samples = samples[len(samples) // 2:]    
    #e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    #print(e_post_mean)
    
    with open(args.data_path, "rb") as f:
        d = json.load(f)
    theta_true = np.array(d["theta"])
    print(theta_true.shape)
    K_truth, V_truth = theta_true.shape[1], theta_true.shape[0] // 2
    if K_truth == K_sample and V_truth == V_sample:
        LOGGER.train(f"Truth K: {K_truth}, V: { V_truth}")
    else:
        LOGGER.error(f"Truth K: {K_truth}, V: { V_truth}, does not match samples")
        exit()
        
    e_truth = Embedding(saved_model_path=str(list(sample_folder.glob("*.pkl"))[-1].absolute()))
    for wd, ix in tqdm.tqdm(list(d["vocabulary"].items())):
        #print(wd, ix)
        wd_c = f"{wd}_c"
        e_truth[wd] = theta_true[ix]
        e_truth[wd_c] = theta_true[ix + V_truth]
    
    #e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    
    rho_true = theta_true[:theta_true.shape[0] // 2]
    alpha_true = theta_true[theta_true.shape[0] // 2:]
    p_true = tf.math.sigmoid(rho_true @ alpha_true.T).numpy()
    
    words = list(sorted(d["vocabulary"], key=lambda wd: d["vocabulary"][wd]))
    contexts = [wd + "_c" for wd in words]
    print(words)
    
    p_samples = np.zeros((len(samples), V_sample, V_sample), dtype=np.float64)
    for ix, sample in tqdm.tqdm(list(enumerate(samples))):
        e_sample = Embedding(saved_model_path=str(sample.absolute()))
        rho_sample = e_sample[words].numpy()
        alpha_sample = e_sample[contexts].numpy()
        assert rho_sample.shape == rho_true.shape
        p_sample = tf.math.sigmoid(rho_sample @ alpha_sample.T).numpy()
        
        assert p_sample.shape == p_true.shape
        p_samples[ix] = p_sample
        #exit()
        #print(rho_sample.shape)

    quantile = (1.0 - args.ci_level) / 2.0
    ci_low_p = np.quantile(p_samples, quantile, axis=0)
    ci_high_p = np.quantile(p_samples, 1.0 - quantile, axis=0)
    #print(ci_low_p)
    #print(ci_high_p)
    p_avg = np.mean(p_samples, axis=0)
    within_ci = (ci_low_p < p_true) * (p_true < ci_high_p)
    #print(within_ci)
    ci_coverage = np.mean(within_ci)
    LOGGER.train(f"CI coverage: {ci_coverage}")
    RMSE_base = np.sqrt(np.mean((p_true - np.mean(p_true)) ** 2))
    LOGGER.train(f"RMSE baseline {RMSE_base}")
    RMSE = np.sqrt(np.mean((p_true - p_avg) ** 2))
    LOGGER.train(f"RMSE: {RMSE}")
    
    # results/5-gibbs-N-10000-D-5-simulation-zwcb/
    folder = Path(args.sample_folder).stem
    dataset_ix = int(folder.split("-")[0])
    N = int(folder.split("-N-")[-1].split("-D-")[0])
    resultdict = {"dataset_ix": dataset_ix, "N": N, "K": K_truth, "V": V_truth, "ci-90-coverage": ci_coverage, "RMSE": RMSE, "RMSE_normalized": RMSE / RMSE_base}
    df = pl.DataFrame(resultdict)
    #print(df)
    coverage_path = Path("logs/coverage.csv")
    if coverage_path.exists():
        old_df = pl.read_csv(coverage_path)
        united_df = pl.concat([old_df, df])
        united_df = united_df.sort("dataset_ix", "N", "K", "V")
        #print(united_df)
        united_df = united_df.unique(["dataset_ix", "N", "K", "V"])
        united_df = united_df.sort("K", "V", "N", "dataset_ix")
        print(united_df)
        united_df.write_csv(coverage_path)
    else:
        print(df)
        df.write_csv(coverage_path)
