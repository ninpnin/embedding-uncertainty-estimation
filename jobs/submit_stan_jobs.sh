#!/usr/bin/env bash

# note!! Output path is dynamically set based on output_path/estimator/dataset
# the idea is to use output_paths like: "results/nofix", "results/fix"
# then easily be able to extract experiments for desired estimator.

# use
#  squeue -u isacbo --Format=jobid,name,partition,state,timeused
# to see ful job name

# ensure fit_models2.py has the correct lambda0




MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../test_optimize/"  #base path
DATASET_DIR="../../new_sims_jan25/test/"
DATALENS=(100)
ESTIMATORS=("hmc") 









MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results-congress/nofix"  #base path
DATASET_DIR="../../us-congress/"
DATALENS=(1000) # remember to set lambda0 = 1.0 in this case!!
ESTIMATORS=("vi") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results-movielens-train/nofix"  #base path
DATASET_DIR="../../movielens_train/"
DATALENS=(85000 170000 425000 850000) # remember to set lambda0 = 1.0 in this case!!
ESTIMATORS=("vi") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results-congress5k/nofix"  #base path
DATASET_DIR="../../us-congress5k/"
DATALENS=(500000) # remember to set lambda0 = 1.0 in this case!!
ESTIMATORS=("vi") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_ten_V200_K10/nofix"  #base path
DATASET_DIR="../../new_sims_jan25/ten_V200_K10/"
DATALENS=( 500000 2000000 4000000)
ESTIMATORS=("hmc") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results-congress5k/nofix"  #base path
DATASET_DIR="../../us-congress5k/"
DATALENS=(100000) # remember to set lambda0 = 1.0 in this case!!
ESTIMATORS=("vi") 





MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_ten_V100K5/nofix/"  #base path
DATASET_DIR="../../data/ten_V100K5/"
DATALENS=(500000 1000000)
ESTIMATORS=("map") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_FULLHMC_V200_K20/nofix"  #base path
DATASET_DIR="../../new_sims_jan25/ten_V200_K20/"
DATALENS=(500000)
ESTIMATORS=("hmc") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_ten_V200_K5/nofix"  #base path
DATASET_DIR="../../new_sims_jan25/ten_V200_K5/"
DATALENS=(1000 2000 5000 10000 20000 50000 100000 500000 1000000)
ESTIMATORS=("hmc") 

MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_ten_V100K5_zipf/nofix/"  #base path
DATASET_DIR="../../data/ten_V100K5-zipf/"
DATALENS=(500000 1000000)
ESTIMATORS=("map") 



MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../test_cmd_hmc/"  #base path
DATASET_DIR="../../new_sims_jan25/test/"
DATALENS=(100)
ESTIMATORS=("cmd_hmc") 


MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results-congress5k/nofix"  #base path
DATASET_DIR="../../us-congress5k/"
DATALENS=(50000 100000 500000) # remember to set lambda0 = 1.0 in this case!!
ESTIMATORS=("map") 


DIMENSION=50 ### IMPORTANT

for DATALEN in "${DATALENS[@]}"; do
    for DATAFILE in "$DATASET_DIR"*.json; do
        for ESTIMATOR in "${ESTIMATORS[@]}"; do
            # create output path based on run settings
            FILENAME=$(basename "$DATAFILE" .json)
            OUTPUT_PATH="$BASE_OUTPUT_PATH/$ESTIMATOR/$FILENAME/"
            mkdir -p "$OUTPUT_PATH" # -p to make sure nothing happens if folder exists.


            SLURM_OUTPUT="${OUTPUT_PATH}/slurm-${DATALEN}-%j.out"
            SLURM_ERROR="${OUTPUT_PATH}/slurm-${DATALEN}-%j.err"

            JOB_NAME="fit_${DATALEN}_$(basename "$DATAFILE" .json)_$ESTIMATOR"


            echo "Submitting job for $DATAFILE with N = $DATALEN, Estimator: $ESTIMATOR, OUTPUT: $OUTPUT_PATH"
            sbatch --output="$SLURM_OUTPUT" --error="$SLURM_ERROR" --export=datafile="$DATAFILE",datalen="$DATALEN",estimator="$ESTIMATOR",model_path="$MODEL_PATH",output_path="$OUTPUT_PATH",dimension="$DIMENSION" -J "$JOB_NAME" stan_sub_job.sh
        done
    done
done
