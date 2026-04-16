#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH -t 21:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --gpus=4
#SBATCH -q regular
#SBATCH -A desi_g
#SBATCH --array=0-3
#SBATCH --output=./slurms/mesh/altmtl-%A_%a.out

source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250526-1.0.0/conda/etc/profile.d/conda.sh
source /global/homes/s/shengyu/env.sh 2pt_env

# MZRR=("False" "repeat" "repeat_zevol" "verr_empirical" "verr_nonparam" "verr_nonparam_zevol")
MZRR=("False" "verr_nonparam")
MOCK_MIN=0
MOCK_MAX=199
MOCKS_PER_ARRAY=100

NUM_ZRR=${#MZRR[@]}
NUM_MOCKS=$((MOCK_MAX - MOCK_MIN + 1))
NUM_BLOCKS=$(((NUM_MOCKS + MOCKS_PER_ARRAY - 1) / MOCKS_PER_ARRAY))
TOTAL_TASKS=$((NUM_ZRR * NUM_BLOCKS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for TOTAL_TASKS=$TOTAL_TASKS"
    exit 1
fi

ZRR_INDEX=$((TASK_ID / NUM_BLOCKS))
BLOCK_INDEX=$((TASK_ID % NUM_BLOCKS))
ZRR="${MZRR[$ZRR_INDEX]}"

MOCK_START=$((MOCK_MIN + BLOCK_INDEX * MOCKS_PER_ARRAY))
MOCK_END=$((MOCK_START + MOCKS_PER_ARRAY - 1))
if (( MOCK_END > MOCK_MAX )); then
    MOCK_END=$MOCK_MAX
fi
MOCK_RANGE="${MOCK_START}-${MOCK_END}"

echo "Node $(hostname) running ZRR=$ZRR mockid=$MOCK_RANGE"
srun -n "${SLURM_NTASKS:-1}" python compute_mesh_jax.py --version holi-v3 --domain altmtl --tracers LRG ELG QSO --mockid "$MOCK_RANGE" --zerrs "$ZRR" --overwrite
