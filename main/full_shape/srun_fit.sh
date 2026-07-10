#!/bin/bash

source /global/homes/s/shengyu/env.sh fit_env

cd "$(dirname "$0")"

srun -N 1 -n 4 -C cpu -c 32 --cpu-bind=cores -t 04:00:00 --qos interactive --exclusive --account desi \
    python run_fits.py \
    --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_errors/fits \
    --version AbacusHF-v2 \
    --domain altmtl \
    --cov_version holi-v3 \
    --tracers QSO1 \
    --zerrs verr_nonparam \
    --stats mesh2 mesh3 \
    --kmax 0.35-0.25-0.10 \
    --mockid 0-24 \
    --cosmo_params base \
    --todos sample \
    --sampler pocomc \
    --resume

# srun -N 1 -n 4 -C cpu -c 32 --cpu-bind=cores -t 04:00:00 --qos interactive --exclusive --account desi \
#     python run_fits.py \
#     --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_errors/fits \
#     --version AbacusHF-v2 \
#     --domain cutsky \
#     --cov_version holi-v3 \
#     --tracers QSO1 \
#     --zerrs  None verr_nonparam \
#     --stats mesh2 \
#     --kmax 0.300-0.300-0.20-0.08 \
#     --mockid 0-24 \
#     --cosmo_params base \
#     --todos sample \
#     --sampler pocomc