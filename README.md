# Posterior Sampling of Probabilistic Word Embeddings

This repository implements the methods presented in _Posterior Sampling of Probabilistic Word Embeddings_, along with the baseline methods of mean-field variational inference and Hamiltonian Monte Carlo.

The required packages are included in ```pyproject.toml```. The most straightforward way is to install the module in ```embedding_uncertainty``` with ```pip install .```. On some systems ```pystan``` needs to be installed in a separate environment, which is why it is not included in ```pyproject.toml```.

All algorithms use the same preprocessed data format. An example is provided in ```data/sim-format.json```.

## SGNS Gibbs sampler

The SGNS Gibbs sampler is implemented in ```embedding_uncertainty/polyagamma_gibbs.py```. This includes a logistic regression posterior sampler and a full Gibbs sampler for SGNS embeddings.

The SGNS Gibbs sampler can be run with ```scripts/gibbs_sampler.py``` :

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

## CBOW Gibbs sampler

The CBOW Gibbs sampler is implemented in ```embedding_uncertainty/polyagamma_gibbs_cbow.py```. This includes a simplistic numpy sampler as well as an optimized TensorFlow sampler.

The CBOW Gibbs sampler can be run with ```scripts/cbow_gibbs_sampler.py``` :

```
usage: cbow_gibbs_sampler.py [-h] [--datapath DATAPATH] [--dim DIM] [--data_len DATA_LEN] [--samples SAMPLES] [--lambda0 LAMBDA0]
                             [--example_word EXAMPLE_WORD] [--prefix PREFIX] [--results_folder RESULTS_FOLDER] [--calculate_p CALCULATE_P]
                             [--benchmark BENCHMARK]

options:
  -h, --help            show this help message and exit
  --datapath DATAPATH
  --dim DIM
  --data_len DATA_LEN
  --samples SAMPLES
  --lambda0 LAMBDA0     Prior strength (variance). If not specified, set to K
  --example_word EXAMPLE_WORD
  --prefix PREFIX
  --results_folder RESULTS_FOLDER
                        Where the samples folder should be placed
  --calculate_p CALCULATE_P
                        Calculate alpha rho.T for ESS etc.
  --benchmark BENCHMARK
                        Only run the script; don't save embeddings
```

## Hamiltonian Monte Carlo and MFVI

Hamiltonian Monte Carlo is implemented in Stan. The unconstrained embedding model is implemented in ```stan_simulation/models/sgns_normalpriors_aggregated.stan```, and the constrained model is implemented in ```stan_simulation/models/sgns_normalpriors_fix_last_aggregated.stan```.

HMC can be run on the unconstrained model with ```stan_simulation/fit_models2.py``` using the ```--estimator hmc``` flag, and MFVI using the ```--estimator vi``` flag

```
usage: fit_models2.py [-h] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR] [--stan_model_path STAN_MODEL_PATH] [--dim DIM] [--lambda0 LAMBDA0]
                      [--estimator ESTIMATOR] [--num_chains NUM_CHAINS] [--num_samples NUM_SAMPLES] [--data_lengths DATA_LENGTHS [DATA_LENGTHS ...]]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --output_dir OUTPUT_DIR
  --stan_model_path STAN_MODEL_PATH
  --dim DIM
  --lambda0 LAMBDA0
  --estimator ESTIMATOR
  --num_chains NUM_CHAINS
  --num_samples NUM_SAMPLES
  --data_lengths DATA_LENGTHS [DATA_LENGTHS ...]
```

and on the constrained model with ```stan_simulation/fit_models2_dxd.py``` using the ```--estimator hmc``` flag, and MFVI using the ```--estimator vi``` flag

```
usage: fit_models2_dxd.py [-h] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR] [--stan_model_path STAN_MODEL_PATH] [--map_dir MAP_DIR] [--dim DIM]
                          [--lambda0 LAMBDA0] [--estimator ESTIMATOR] [--num_chains NUM_CHAINS] [--num_samples NUM_SAMPLES]
                          [--data_lengths DATA_LENGTHS [DATA_LENGTHS ...]]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --output_dir OUTPUT_DIR
  --stan_model_path STAN_MODEL_PATH
  --map_dir MAP_DIR     Directory containing the previously fitted unfixed MAP models.
  --dim DIM
  --lambda0 LAMBDA0
  --estimator ESTIMATOR
  --num_chains NUM_CHAINS
  --num_samples NUM_SAMPLES
  --data_lengths DATA_LENGTHS [DATA_LENGTHS ...]
```

## Preprocessing

A simulated dataset can be created using ```simulate_data/simulate.py```. Based on such a dataset, a Zipf distributed dataset can be generated using ```scripts/zipf_simulation.py```.

The MovieLens data was preprocessed using ```real_data/movielens/analyze_data.ipynb```.

The US congress data was preprocessed using ```scripts/us-congress-convert-and-lemmatize.py``` and ```scripts/us-congress-create-sgns-data.py```.

## Jobs

The ```jobs/``` folder contains scripts and slurm job files that run the experiments in the article.

## Logs

Logs and aggregated results of the experiments are saved in the ```logs/``` folder.

## Plots

Different plots in the article are generated with the following scripts

```
plots/alternative_plots/applied_plots.ipynb
plots/convergence.ipynb
plots/donut_plots.ipynb
plots/rmse_convergence.ipynb
scripts/gibbs_hmc_performance_plot.py
scripts/map_pm_posterior_plot.py
scripts/rmse_plot.py
scripts/wordsim_cis.py
```

## Tests

The Polya-Gamma sampler and related functions are tested via the ```unittest``` Python module in the ```tests``` folder.