#!/usr/bin/env bash

MODEL_PATH="../stan_simulation/models/sgns_normalpriors_fix_last_aggregated.stan"
#MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
#MODEL_PATH="../stan_simulation/models/sgns_dxd_aggregated.stan"

## -- test
BASE_OUTPUT_PATH="../../test_fix_res/fix"  
BASE_MAP_DIR="../../test_fix_res/nofix/map"        # Where the unfixed MAP models are stored. 
# Example: "../../test_fix_res/nofix/map" where map/ containt folder 1, 2, ... each containing map_something_{sizes}.pkl
DATASET_DIR="../../test_100k5/"
DATALENS=(1000) 
ESTIMATORS=("hmc")   
DIMENSION=5
## --


## ---
BASE_OUTPUT_PATH="../../results50k_V100K5/fix"  
BASE_MAP_DIR="../../results_ten_V100K5/nofix/map"        # Where the unfixed MAP models are stored. 
# Example: "../../test_fix_res/nofix/map" where map/ containt folder 1, 2, ... each containing map_something_{sizes}.pkl
DATASET_DIR="../../data/ten_V100K5/"
#DATASET_DIR="../../new_sims_jan25/ten_V200_K10/"

DATALENS=(50000) 
ESTIMATORS=("vi")              

DIMENSION=5 # -------------------------------<<

BASE_OUTPUT_PATH="../../test_fix_res/fix"  
BASE_MAP_DIR="../../test_fix_res/nofix/map"        # Where the unfixed MAP models are stored. 
# Example: "../../test_fix_res/nofix/map" where map/ containt folder 1, 2, ... each containing map_something_{sizes}.pkl
DATASET_DIR="../../test_100k5/"
DATALENS=(1000) 
ESTIMATORS=("vi")   
DIMENSION=5



#MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_FULLHMC_V100_K5/fix"  #base path
DATASET_DIR="../../one_V100_K5/"
BASE_MAP_DIR="../../results_ten_V100K5/nofix/map" 
DATALENS=(50000)
ESTIMATORS=("hmc") 



## ---- test
MODEL_PATH="../stan_simulation/models/sgns_normalpriors_fix_last_aggregated.stan"
BASE_OUTPUT_PATH="../../test_fix_res/fix"  
BASE_MAP_DIR="../../test_fix_res/nofix/map"        # Where the unfixed MAP models are stored. 
# Example: "../../test_fix_res/nofix/map" where map/ containt folder 1, 2, ... each containing map_something_{sizes}.pkl
DATASET_DIR="../../test_100k5/"
DATALENS=(1000) 
ESTIMATORS=("hmc")   
DIMENSION=3


# --- congress 5k
MODEL_PATH="../stan_simulation/models/sgns_normalpriors_fix_last_aggregated.stan"
BASE_OUTPUT_PATH="../../results-congress5k/fix"  #base path
DATASET_DIR="../../us-congress5k/"
BASE_MAP_DIR="../../results-congress5k/nofix/map"  # assumes dataname as inner folder name
DATALENS=(50000)
ESTIMATORS=("hmc") 
DIMENSION=50



## ---- test
MODEL_PATH="../stan_simulation/models/sgns_normalpriors_fix_last_aggregated.stan"
BASE_OUTPUT_PATH="../../test_fix_res/fix"  
BASE_MAP_DIR="../../test_fix_res/nofix/map"        # Where the unfixed MAP models are stored. 
# Example: "../../test_fix_res/nofix/map" where map/ containt folder 1, 2, ... each containing map_something_{sizes}.pkl
DATASET_DIR="../../test_100k5/"
DATALENS=(1000) 
ESTIMATORS=("hmc")   
DIMENSION=3


for DATALEN in "${DATALENS[@]}"; do
    for DATAFILE in "$DATASET_DIR"*.json; do
        for ESTIMATOR in "${ESTIMATORS[@]}"; do

            FILENAME=$(basename "$DATAFILE" .json) ## 1.json -> "1"
            OUTPUT_PATH="$BASE_OUTPUT_PATH/$ESTIMATOR/$FILENAME/"  # save_folder/hmc/1
            MAP_DIR="$BASE_MAP_DIR/$FILENAME/" ## map_folder/1/"
            mkdir -p "$OUTPUT_PATH"

            SLURM_OUTPUT="${OUTPUT_PATH}/slurm-${DATALEN}-%j.out"
            SLURM_ERROR="${OUTPUT_PATH}/slurm-${DATALEN}-%j.err"

            JOB_NAME="fit_${DATALEN}_${FILENAME}_${ESTIMATOR}"

            echo "Submitting job for $DATAFILE with N = $DATALEN, Estimator: $ESTIMATOR"
            echo "   Output path: $OUTPUT_PATH"
            sbatch --output="$SLURM_OUTPUT" \
                   --error="$SLURM_ERROR" \
                   --export=datafile="$DATAFILE",\
datalen="$DATALEN",\
estimator="$ESTIMATOR",\
model_path="$MODEL_PATH",\
output_path="$OUTPUT_PATH",\
dimension="$DIMENSION",\
map_dir="$MAP_DIR",\
lambda0="$LAMBDA0" \
                   -J "$JOB_NAME" \
                   stan_sub_job_fix.sh
        done
    done
done
