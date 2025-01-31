#!/usr/bin/env bash

MODEL_PATH="../stan_simulation/models/sgns_dxd_aggregated_lastrows.stan"

BASE_OUTPUT_PATH="../../test_fix_res/fix"  
BASE_MAP_DIR="../../test_fix_res/nofix/map"        # Where the unfixed MAP models are stored. 
# Example: "../../test_fix_res/nofix/map" where map/ containt folder 1, 2, ... each containing map_something_{sizes}.pkl
DATASET_DIR="../../test_100k5/"


DATALENS=(1000) 
ESTIMATORS=("hmc")              


DIMENSION=5
# LAMBDA0=1.0  # optional 

for DATALEN in "${DATALENS[@]}"; do
    for DATAFILE in "$DATASET_DIR"*.json; do
        for ESTIMATOR in "${ESTIMATORS[@]}"; do

            FILENAME=$(basename "$DATAFILE" .json) ## 1.json -> "1"
            OUTPUT_PATH="$BASE_OUTPUT_PATH/$ESTIMATOR/$FILENAME/"  # save_folder/hmc/1
            MAP_DIR="BASE_MAP_DIR/$FILENAME/" ## map_folder/1/"
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
