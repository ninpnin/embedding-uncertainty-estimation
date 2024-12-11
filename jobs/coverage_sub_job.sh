#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-1419
#SBATCH -J coverage_${experiment_id}_${estimator}
#SBATCH -t 0-03:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p shared

ml PDC/23.12
ml anaconda3/2024.02-1-cpeGNU-23.12

DATA_FOLDER="$data_folder"
FIT_FOLDER="$fit_folder"
OUTPUT_PATH="$output_path"
EXPERIMENT_ID="$experiment_id"
ESTIMATOR="$estimator"
FIT_FOLDER_BASE_NAME="$fit_folder_base_name"
BASE_DATA_NAME="$base_data_name"

PYTHON_SCRIPT="../plots/calculate_coverage.py"

echo "Experiment ID: $EXPERIMENT_ID, Estimator: $ESTIMATOR"
echo "FIT_FOLDER_BASE_NAME: $FIT_FOLDER_BASE_NAME, BASE_DATA_NAME: $BASE_DATA_NAME"

python "$PYTHON_SCRIPT" \
    --data_folder "$DATA_FOLDER" \
    --fit_folder "$FIT_FOLDER" \
    --output_path "$OUTPUT_PATH" \
    --experiment_id "$EXPERIMENT_ID" \
    --estimator "$ESTIMATOR" \
    --fit_folder_base_name "$FIT_FOLDER_BASE_NAME" \
    --base_data_name "$BASE_DATA_NAME"
