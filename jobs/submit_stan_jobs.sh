#!/usr/bin/env bash

# note!! Output path is dynamically set based on output_path/estimator/dataset
# the idea is to use output_paths like: "results/nofix", "results/fix"
# then easily be able to extract experiments for desired estimator.

# use
#  squeue -u isacbo --Format=jobid,name,partition,state,timeused
# to see ful job name

MODEL_PATH="../stan_simulation/models/sgns_normalpriors_aggregated.stan"
BASE_OUTPUT_PATH="../../results_ten_V100K5/nofix"  #base path
DATASET_DIR="../../data/ten_V100K5/"
DATALENS=(100 200)
ESTIMATORS=("hmc vi") 

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
            sbatch --output="$SLURM_OUTPUT" --error="$SLURM_ERROR" --export=datafile="$DATAFILE",datalen="$DATALEN",estimator="$ESTIMATOR",model_path="$MODEL_PATH",output_path="$OUTPUT_PATH" -J "$JOB_NAME" stan_sub_job.sh
        done
    done
done
