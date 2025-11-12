#!/bin/bash

# Activate environments
activate_env() {
    case $1 in
        desi)
            source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            ;;
        2pt)
            module load GCC/9.3.0  OpenMPI/4.0.3  FFTW/3.3.8
            source /opt/ebsofts/Anaconda3/2024.02-1/etc/profile.d/conda.sh
            conda activate ~/.conda/envs/2pt_env
            ;;
    esac
}

# Run srun command
run_srun() {
    case $1 in
        2pt)
            srun -n 1 -c 64 -t 04:00:00 -p public-cpu python compute_statistics.py  --tracers 'LRG' 'ELG' 'QSO' --mockid "0-24"
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