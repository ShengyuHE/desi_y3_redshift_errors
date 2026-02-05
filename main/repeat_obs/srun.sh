#!/bin/bash

# Function to activate environments
activate_environment(){
    case $1 in
        get_repeat_AN | get_repeat | variance | repeat_model)
            source /global/homes/s/shengyu/env.sh rc_env
            ;;
    esac
}

# Function to run srun command
run_srun() {
    case "$1" in
        repeat_AN)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi \
                 python desi_main_repeats.py \
                    --outroot /pscratch/sd/s/shengyu/repeats/DA2/kibo-v1/ \
                    --prod kibo \
                    --prog bright \
                    --steps parent,pairs,plot \
                    --numproc 8 \
                    --overwrite
            ;;
        get_repeat)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python get_repeat_redshifts.py
            ;;
        variance)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python repeats_variance.py 
            ;;
        repeat_model)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python model_repeats.py --tracer BGS --ztype global bin
            ;;
    esac
}

# Require an argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 AN_repeats"
    exit 1
fi

activate_environment "$1"
run_srun "$1"
