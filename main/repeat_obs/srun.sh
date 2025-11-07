#!/bin/bash

# Function to activate environments
activate_environment(){
    case $1 in
        cp | add_dv)
            source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            ;;
    esac
}

# Function to run srun command
run_srun() {
    case "$1" in
        AN_repeats)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos=interactive --account=desi \
                 python desi_main_repeats.py \
                    --outroot /pscratch/sd/s/shengyu/repeats/DA2/kibo-v1/ \
                    --prod kibo \
                    --prog bright \
                    --steps parent,pairs,plot \
                    --numproc 8 \
                    --overwrite
            ;;
        *)
            echo "Error: unknown mode '$1'." >&2
            echo "Usage: $0 AN_repeats" >&2
            exit 1
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
