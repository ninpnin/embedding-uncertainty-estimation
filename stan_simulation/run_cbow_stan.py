import os, json, numpy as np
from cmdstanpy import CmdStanModel
from bidict import bidict

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
    args = parser.parse_args()


    N, K = args.data_len, args.K
    datafile = args.datapath #f'../tests/data/data-cbow-K-{K}-V-{V}-N-100000.json'
    #TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), datafile)
    with open(datafile) as f:
        docs = json.load(f)

    # --- Data setup ---  NOTE! indexing starts on 1 with stan, so the vocab will be different from numpy Gibbs sampler.
    N = 1000

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

    vocab = ww + ([f'{w_i}_c' for w_i in ww])
    vocab = set(vocab) 
    V = len(vocab)

    word2id = bidict({w: i+1 for i, w in enumerate(vocab)})#   +1 <<<------

    w_idx = np.array([word2id[w] for w in ww], dtype=int)
    C_idx = np.array([[word2id[v + "_c"] for v in C_i] for C_i in CC])
    x_vec = np.array(xx, dtype=int)


    # --- Stan Setup ---

    stan_file = "cbow.stan"


    WS = len(CC[0]) 


    stan_data = {
        "s": 1.0,  # TODO: 1/sqrt(lambda)
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

    #print("")
