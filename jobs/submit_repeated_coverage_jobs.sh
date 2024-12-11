#!/usr/bin/env bash

# observe data sizes are hardcoded in calculate_coverage :/

BASE_FIT_FOLDER="../../results_ten_V100K5_zipf/nofix"
BASE_OUTPUT_PATH="../../coverage_results_ten_V100K5_zipf" 
DATASET_DIR="../../data/ten_V100K5-zipf"
ESTIMATOR="vi"  

EXPERIMENT_IDS=(1 2 3 4 5 6 7 8 9 10) 
# example: "%i-zipf"%experiment_id
FIT_FOLDER_BASE_NAME="%i-zipf"
BASE_DATA_NAME="%i-zipf.json"

for EXPERIMENT_ID in "${EXPERIMENT_IDS[@]}"; do

    FIT_FOLDER="$BASE_FIT_FOLDER" #/$EXPERIMENT_ID/"
    OUTPUT_PATH="$BASE_OUTPUT_PATH" #/$EXPERIMENT_ID/"
    mkdir -p "$OUTPUT_PATH"  # Ensure the folder exists for coverage results

    # does not put in the exact correct folder...
    SLURM_OUTPUT="${OUTPUT_PATH}/slurm-${EXPERIMENT_ID}-%j.out"
    SLURM_ERROR="${OUTPUT_PATH}/slurm-${EXPERIMENT_ID}-%j.err"

    JOB_NAME="coverage_${EXPERIMENT_ID}_${ESTIMATOR}"

    echo "Submitting job for repetition = $EXPERIMENT_ID, FIT_FOLDER: $FIT_FOLDER, OUTPUT: $OUTPUT_PATH"
    sbatch --output="$SLURM_OUTPUT" --error="$SLURM_ERROR" \
        --export=data_folder="$DATASET_DIR",fit_folder="$FIT_FOLDER",output_path="$OUTPUT_PATH",experiment_id="$EXPERIMENT_ID",estimator="$ESTIMATOR",fit_folder_base_name="$FIT_FOLDER_BASE_NAME",base_data_name="$BASE_DATA_NAME" \
        -J "$JOB_NAME" coverage_sub_job.sh
done
