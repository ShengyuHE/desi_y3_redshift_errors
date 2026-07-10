#!/bin/bash

# Activate environments
activate_env() {
    case $1 in
        add_zerr | convert )
            # source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            source /global/homes/s/shengyu/env.sh rc_env
            ;;
    esac
}

## zerr: None repeat verr_empirical verr_nonparam
# Run srun command
run_srun() {
    case $1 in
        # add_zerr)
            # srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python build_zerr_mocks.py  --version AbacusHF-v2  --tracer LRG --domain cubic --zerrs None repeat verr_empirical verr_nonparam
            # ;;
        convert)
            srun -N 1 -n 4 -c 32 -C cpu -t 04:00:00 --qos interactive --account desi python convert_mocks.py --version AbacusHF-4snap --domain lightcone --mockid 1-24 --hod base
            # srun -N 1 -n 4 -c 32 -C cpu -t 04:00:00 --qos interactive --account desi python convert_mocks.py --version AbacusHF-v2 --domain cutsky --mockid 1-24 --hod base_dv 
            ;;
    esac
}

# Check if a computation type was provided
if [ -z "$1" ]; then
    echo "Usage: bash srun.sh [add_zerr|convert]"
    exit 1
fi

# Run the srun command
activate_env $1
run_srun $1
