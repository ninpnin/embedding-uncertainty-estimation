import os, json, numpy as np
from cmdstanpy import CmdStanModel
from bidict import bidict
import tqdm
import random, string
from pathlib import Path
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../tests/data/data-cbow-K-10-V-100-N-100000.json")
    parser.add_argument("--K", type=int, default=10, help="Dimensionality of the embeddings")
    parser.add_argument("--S", type=int, default=2, help="S hyperparameter for the PG-Gibbs algorithm")
    parser.add_argument("--data_len", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--lambda0", type=float, default=None, help="Prior strength (variance). If not specified, set to K [TODO]")
    parser.add_argument("--results_folder", type=str, default="../results", help="Where the samples folder should be placed")
    args = parser.parse_args()


    N, K = args.data_len, args.K
    datafile = args.datapath #f'../tests/data/data-cbow-K-{K}-V-{V}-N-100000.json'
    #TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), datafile)
    with open(datafile) as f:
        docs = json.load(f)

    # --- Data setup ---  NOTE! indexing starts on 1 with stan, so the vocab will be different from numpy Gibbs sampler.
    N = args.data_len

    vocab = set()
    # Aggregate data into matrices
    ww = []
    CC = []
    xx = []
    for elem in docs[:N]:
        w_i, C_i, x_i = elem["w"], elem["C"], elem["x"]
        ww.append(w_i)
        CC.append(C_i)
        xx.append(x_i)

    vocab = set(ww)
    V = len(vocab)

    word2id = bidict({wd: ix + 1 for ix, wd in enumerate(vocab)}) #   +1 <<<------

    w_idx = np.array([word2id[w] for w in ww], dtype=int)
    C_idx = np.array([[word2id[v] for v in C_i] for C_i in CC])
    x_vec = np.array(xx, dtype=int)

    word2id_numpy = bidict({wd: word2id[wd] - 1 for wd in word2id})

    # --- Stan Setup ---

    stan_file = "cbow.stan"


    WS = len(CC[0]) 

    # --- SAMPLER SETUP ----
    lam = K
    if args.lambda0 is not None:
        lam = args.lambda0
        LOGGER.info(f"Set lambda0 from argparse parameters {lam}")
    else:
        LOGGER.info(f"Set lambda0 to default sqrt(K) = {lam}")

    s = 1/np.sqrt(lam)
    LOGGER.info(f"This implies a prior N(0, s^2) = N(0, {s ** 2}")

    stan_data = {
        "s": s,  # TODO: 1/sqrt(lambda)
        "V": V,
        "K": K,
        "N": N,
        "WS": WS,
        "target_word": w_idx.tolist(),
        "context_idx": C_idx.tolist(),
        "posneg_labels": x_vec.tolist(),
    }

    # --- Model ---
    stan_file = "cbow.stan"

    model = CmdStanModel(stan_file=f"models/{stan_file}")
    fit = model.sample(
        data=stan_data,
        chains=args.chains,
        iter_warmup=args.n_samples,
        iter_sampling=args.n_samples,
        seed=1,
        show_console=True
    )

    print(fit.summary())

    rho_samples = fit.stan_variable("rho")
    alpha_samples = fit.stan_variable("alpha")

    words = [word2id_numpy.inv[ix] for ix in range(V)]
    contexts = [wd + "_c" for wd in words]

    pathstem = Path(args.datapath).stem.replace("_", "-")
    randomchars = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    samples_folder = f"{pathstem}-cbow-stan-N-{N}-K-{K}-V-{V}-{randomchars}"
    results_folder = Path(args.results_folder)
    results_folder.mkdir(exist_ok=True)
    samples_folder = (results_folder / samples_folder)
    samples_folder.mkdir(exist_ok=True)

    # Do the pwe import at this point so that the script can be run without tensorflow
    from probabilistic_word_embeddings.embeddings import Embedding
    for sample_ix, emb in tqdm.tqdm(enumerate(zip(rho_samples, alpha_samples))):

        rho, alpha = emb
        e_sample = Embedding(set(vocab), dimensionality=K)
        e_sample[words] = rho
        e_sample[contexts] = alpha

        sample_path = samples_folder / f"sample-{sample_ix}.json"
        sample_path_str = str(sample_path.resolve())
        e_sample.save(sample_path_str)
