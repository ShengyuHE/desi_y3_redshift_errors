#!/bin/bash

source /global/homes/s/shengyu/env.sh fit_env

cd "$(dirname "$0")"

srun -N 1 -n 4 -C cpu -t 04:00:00 --qos interactive --exclusive --account desi \
    python run_fits.py \
    --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_errors/fits \
    --version AbacusHF-v2 \
    --cov_version holi-v3 \
    --tracers LRG1 \
    --zerrs  None verr_nonparam \
    --stats mesh2 \
    --mockid 0-24 \
    --cosmo_params base \
    --todos sample \
    --sampler pocomc
