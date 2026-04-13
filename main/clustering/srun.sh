#!/bin/bash

# Activate environments
activate_env() {
    case $1 in
        cat )
            # source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            source /global/homes/s/shengyu/env.sh rc_env
            ;;
        mesh | 2pt)
            # source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250526-1.0.0/conda/etc/profile.d/conda.sh
            source /global/homes/s/shengyu/env.sh 2pt_env           
            ;;
    esac
}

## zerr: None repeat verr_empirical verr_nonparam
# Run srun command
run_srun() {
    case $1 in
        cat)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python build_catalogs.py  --version AbacusHF-v2  --tracer LRG --domain cubic --zerrs None repeat verr_empirical verr_nonparam
            # srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python build_catalogs.py  --version AbacusHF-v1  --tracer LRG --domain cutsky --zerrs None --mockid 0
            ;;
        2pt)
            srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_2pt.py --version AbacusHF-v2  --tracers LRG --domain cubic --zerrs verr_nonparam
            # srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_2pt.py --version AbacusHF-v1  --tracers LRG --domains cutsky --zerrs None --mockid 0
            ;;
        mesh)
            srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --region NGC SGC GCcomb --tracers QSO --domain altmtl --zerrs None repeat repeat_zevol verr_empirical verr_nonparam verr_nonparam_zevol --todo mesh2
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version AbacusHF-v1  --tracers LRG --domains cutsky --zerrs None --mockid 0 --todo mesh2
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