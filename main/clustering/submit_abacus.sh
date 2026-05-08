#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -t 17:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -A desi_g
#SBATCH --chdir=/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering
#SBATCH --output=/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering/slurm_logs/Abacus_%j.out

set -euo pipefail

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MPIIO_DVS_MAXNODES=1

ZERRS=("False" "repeat" "verr_nonparam" "verr_nonparam_zevol")

status=0
for zerr in "${ZERRS[@]}"; do
    echo "Launching AbacusHF-v2 bispectrum zerr=${zerr} on one node for default mock range and tracers LRG ELG QSO"
    srun --exclusive -N 1 -n 4 --gpus-per-node=4 python compute_mesh_jax.py \
        --version AbacusHF-v2 \
        --domain altmtl \
        --tracers LRG ELG QSO \
        --zerrs "${zerr}" \
        --todos mesh3_sugiyama
done

if ! wait; then
    status=1
fi
exit "${status}"
