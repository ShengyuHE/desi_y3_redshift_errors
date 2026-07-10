#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH -t 11:00:00
#SBATCH -C cpu
#SBATCH --array=0-5
#SBATCH -q regular
#SBATCH -A desi
#SBATCH --output=/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/full_shape/slurms/logs/mesh3_bk000_%A_%a.out

set -euo pipefail

source /global/homes/s/shengyu/env.sh fit_env
export MPICH_MPIIO_DVS_MAXNODES=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

SCRIPT_DIR='/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/full_shape'

TRACERS=('LRG1' 'LRG2' 'LRG3')
ZERRS=('None' 'verr_nonparam')

NTRACERS=${#TRACERS[@]}
NZERRS=${#ZERRS[@]}
TOTAL_TASKS=$((NTRACERS * NZERRS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID} for TOTAL_TASKS=${TOTAL_TASKS}"
    exit 1
fi

TRACER_INDEX=$((TASK_ID / NZERRS))
ZERR_INDEX=$((TASK_ID % NZERRS))

tracer=${TRACERS[$TRACER_INDEX]}
zerr=${ZERRS[$ZERR_INDEX]}

cd "${SCRIPT_DIR}"

echo "Node $(hostname) running run_fits.py tracer=${tracer} zerr=${zerr} stats=mesh2 mockid=0-24"
srun -N 1 -n 4 -C cpu -c "${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores python -u "${SCRIPT_DIR}/run_fits.py" \
    --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_errors/fits \
    --version AbacusHF-v2 \
    --domain altmtl \
    --hod base \
    --cov_version holi-v3 \
    --tracers "$tracer" \
    --zerrs  "$zerr" \
    --stats mesh2 mesh3 \
    --kmax 0.35-0.25-0.20-0.08 \
    --mockid 0-24 \
    --cosmo_params base \
    --todos sample \
    --sampler pocomc