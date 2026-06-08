#!/bin/bash

source /global/homes/s/shengyu/env.sh fit_env

cd "$(dirname "$0")"

# kmaxs=(0.300-0.300-0.20-0.08 0.250-0.250-0.20-0.08 0.200-0.200-0.20-0.08)
kmaxs=(0.300-0.300-0.20-0.08 0.300-0.300-0.20-0.00 0.300-0.300-0.12-0.08 0.300-0.300-0.12-0.00)

for kmax in "${kmaxs[@]}"; do
    srun -N 1 -n 4 -C cpu -t 04:00:00 --qos interactive --exclusive --account desi \
        python run_fits.py \
        --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/test_2Gpc/fits \
        --version AbacusHF-test \
        --cov_version EZmocks-test \
        --cov_scale 27.0 \
        --domain cubic \
        --tracers QSO1 \
        --stats mesh2 mesh3 \
        --kmax $kmax \
        --mockid 0-24 \
        --theory_model folpsD \
        --cosmo_params base \
        --todos sample \
        --sampler pocomc
done
