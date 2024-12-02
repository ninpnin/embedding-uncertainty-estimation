#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-1419
#SBATCH -J fit_${datafile##*/}_${datalen}_${estimator}
#SBATCH -t 0-03:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p shared

ml PDC/23.12
ml anaconda3/2024.02-1-cpeGNU-23.12
# source conda.init.sh

DATAFILE="$datafile"
DATALEN="$datalen"
ESTIMATOR="$estimator"
MODEL_PATH="$model_path"
OUTPUT_PATH="$output_path"

echo "Running $DATAFILE with N = $DATALEN, Estimator: $ESTIMATOR"
LOGLEVEL=DEBUG python ../stan_simulation/fit_models2.py --data_path "$DATAFILE" \
    --stan_model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --data_lengths "$DATALEN" \
    --estimator "$ESTIMATOR"
