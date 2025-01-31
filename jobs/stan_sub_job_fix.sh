#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-1419
#SBATCH -J fit_${datafile##*/}_${datalen}_${estimator}
#SBATCH -t 0-01:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p shared

# Load your modules
ml PDC/23.12
ml anaconda3/2024.02-1-cpeGNU-23.12

# Slurm environment variables to Python
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

DATAFILE="$datafile"
DATALEN="$datalen"
ESTIMATOR="$estimator"
MODEL_PATH="$model_path"
OUTPUT_PATH="$output_path"
DIMENSION="$dimension"
MAP_DIR="$map_dir"        # <-- Fix dir
LAMBDA0="$lambda0"           

echo "Running $DATAFILE with N = $DATALEN, Estimator: $ESTIMATOR, lambda0: $LAMBDA0"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" \
    /pdc/software/23.12/eb/software/anaconda3/2024.02-1-cpeGNU-23.12/bin/python ../stan_simulation/fit_models2_dxd.py \
        --data_path "$DATAFILE" \
        --stan_model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_PATH" \
        --data_lengths "$DATALEN" \
        --estimator "$ESTIMATOR" \
        --dim "$DIMENSION" \
        --map_dir "$MAP_DIR" \
        --lambda0 "$LAMBDA0"
