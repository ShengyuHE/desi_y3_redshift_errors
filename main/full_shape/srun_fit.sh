#!/bin/bash

source /global/homes/s/shengyu/env.sh fit_env

srun -N 1 -n 4 -C cpu -t 04:00:00 --qos interactive --exclusive --account desi \
    python run_fits.py \
    --tracers LRG3 \
    --zerrs  None verr_nonparam \
    --stats mesh2 \
    --mockid 0-24 \
    --todos sample \
    --sampler pocomc