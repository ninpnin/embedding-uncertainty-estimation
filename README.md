# Posterior Sampling of Probabilistic Word Embeddings

This repository implements the methods presented in _Posterior Sampling of Probabilistic Word Embeddings_, along with the baseline methods of mean-field variational inference and Hamiltonian Monte Carlo.

The required packages are included in ```pyproject.toml```. The most straightforward way is to install the module in ```embedding_uncertainty``` with ```pip install .```.

## Gibbs sampler

The Gibbs sampler is implemented in ```embedding_uncertainty/polyagamma_gibbs.py```. This includes a logistic regression posterior sampler and a full Gibbs sampler for SGNS embeddings.

The Gibbs sampler can be run with ```scripts/gibbs_sampler.py``` :

```
usage: gibbs_sampler.py [-h] [--datapath DATAPATH] [--shuffle_data SHUFFLE_DATA] [--dim DIM] [--data_len DATA_LEN] [--samples SAMPLES]
                        [--map_estimate MAP_ESTIMATE] [--lambda0 LAMBDA0] [--example_word EXAMPLE_WORD] [--use_tf USE_TF] [--prefix PREFIX]
                        [--pg_iter PG_ITER] [--mvn_method {cholesky,svd}] [--calculate_p CALCULATE_P] [--plot PLOT]

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH
  --shuffle_data SHUFFLE_DATA
  --dim DIM
  --data_len DATA_LEN
  --samples SAMPLES
  --map_estimate MAP_ESTIMATE
  --lambda0 LAMBDA0     Prior strength (variance). If not specified, set to K
  --example_word EXAMPLE_WORD
  --use_tf USE_TF
  --prefix PREFIX
  --pg_iter PG_ITER
  --mvn_method {cholesky,svd}
  --calculate_p CALCULATE_P
  --plot PLOT
```

## Laplace approximation

The Gibbs sampler is implemented in ```embedding_uncertainty/laplace_approx.py```.

The Laplace approximation can be run on the simulated data with ```scripts/laplace_approximation.py``` :

```
usage: laplace_approximation.py [-h] [--embedding EMBEDDING] [--datapath DATAPATH] [--data_len DATA_LEN] [--word WORD] [--context CONTEXT] [--samples SAMPLES]
                                [--ci_alpha CI_ALPHA] [--elementwise ELEMENTWISE] [--save_folder SAVE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --embedding EMBEDDING
  --datapath DATAPATH
  --data_len DATA_LEN
  --word WORD
  --context CONTEXT
  --samples SAMPLES
  --ci_alpha CI_ALPHA
  --elementwise ELEMENTWISE
  --save_folder SAVE_FOLDER
```

