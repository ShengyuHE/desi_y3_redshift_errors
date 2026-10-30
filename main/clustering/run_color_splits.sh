#!/bin/bash
# QSO mock 0: compute NGC/SGC prerequisites, then combine them into GCcomb.
set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
unset JAX_PLATFORMS JAX_PLATFORM_NAME
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_MPIIO_DVS_MAXNODES=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "${BASH_SOURCE[0]}")"
srun -N 1 -n 4 -C 'gpu&hbm80g' -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
    python compute_mesh_jax.py --version AbacusHF-v2 --tracers QSO --domain cutsky \
    --hod base_dv --mockid 0 --zerrs repeat_blue repeat_red repeat_all --regions NGC SGC GCcomb --todos mesh2
