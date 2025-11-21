#!/bin/bash

# Activate environments
activate_env() {
    case $1 in
        desi | cat | 2pt)
            source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            ;;
    esac
}

# Run srun command
run_srun() {
    case $1 in
        cat)
            srun -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python build_catalogs.py --domain cutsky --tracer QSO
            ;;
        2pt)
            srun -n 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi python compute_statistics.py --task 2pt --tracer ELG
            ;;
    esac
}

# Check if a computation type was provided
if [ -z "$1" ]; then
    echo "Usage: ./srun_combined.sh [pk|fs]"
    exit 1
fi

# Run the srun command
activate_env $1
run_srun $1