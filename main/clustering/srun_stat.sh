#!/bin/bash

# Activate environments
activate_env() {
    case $1 in
        mesh | 2pt | debug | window)
            source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            unset JAX_PLATFORMS
            unset JAX_PLATFORM_NAME
            export MPICH_GPU_SUPPORT_ENABLED=1
            export MPICH_MPIIO_DVS_MAXNODES=1
            # source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250526-1.0.0/conda/etc/profile.d/conda.sh
            # source /global/homes/s/shengyu/env.sh 2pt_env           
            ;;
    esac
}

## zerr: None repeat verr_empirical verr_nonparam
# Run srun command
run_srun() {
    case $1 in
        2pt)
            srun -N 1 -n 4 -C cpu -c 32 -t 04:00:00 --qos interactive --account desi python -u compute_2pt.py --version AbacusHF-v2 --tracers QSO --regions NGC SGC ALL --domain cutsky --mockid 0-24 --hod base_dv --zerrs None verr_nonparam --nthreads 32
            # srun -N 1 -n 4 -C cpu -c 32 -t 04:00:00 --qos interactive --account desi python -u compute_2pt.py --version AbacusHF-v2 --tracers LRG --regions NGC SGC ALL --domain altmtl --mockid 0-24 --zerrs None verr_nonparam --nthreads 32 --task wplog
            # srun -N 1 -n 4 -C cpu -c 32 -t 04:00:00 --qos interactive --account desi python -u compute_2pt.py --version holi-v3 --tracers LRG QSO --regions NGC SGC ALL --domain altmtl --mockid 0-120 --zerrs None --nthreads 32 --task wplog
            ;;
        mesh)
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --exclusive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers LRG ELG QSO --domain altmtl --zerrs None repeat verr_nonparam verr_nonparam_zevol --regions NGC SGC ALL --todos mesh3_sugiyama
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version AbacusSecondGen --tracers BGS --domain altmtl --mockid 0-24 --zerrs None repeat verr_nonparam verr_nonparam_zevol --todos mesh2
            srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers ELG --domain altmtl --mockid 0-24 --zerrs repeat verr_nonparam --todos mesh3_sugiyama
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version data-dr1-v1.5 --tracers QSO --domain altmtl --ntile_cut 0 --todos mesh2
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --exclusive --account desi_g python compute_mesh_jax.py --version AbacusHF-4snap --tracers QSO --domain lightcone --zranges "0.8-2.1" --mockid 0-24 --todos mesh2
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --exclusive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers QSO --domain cutsky --hod base_dv --regions NGC SGC ALL --mockid 0 --zerrs verr_nonparam --todos mesh3_sugiyama
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --exclusive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers QSO --domain cutsky --hod base_dv --regions NGC SGC ALL --mockid 0-24 --zerrs None verr_nonparam verr_nonparam_zevol repeat --todos mesh3_sugiyama
            # -zerr None repeat repeat_zevol verr_empirical verr_nonparam verr_nonparam_zevol
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version holi-v3 --tracers ELG --domain altmtl --zerrs None --mockid 0-999 --todos mesh2
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version holi-v3 --tracers QSO --regions NGC SGC ALL --domain altmtl --zerrs None --mockid 0-10 --todos mesh2 mesh3_sugiyama
            ;;
        window)
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers LRG --domain altmtl --mockid 0-24 --zerrs repeat verr_nonparam verr_nonparam_zevol --todos mesh2_window
            srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --exclusive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers QSO --domain cutsky --hod base_dv --regions NGC SGC ALL --mockid 0 --zerrs verr_nonparam --todos mesh3_sugiyama_window
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --exclusive --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers QSO --domain cutsky --hod base_dv --regions NGC SGC ALL --mockid 0-24 --zerrs None verr_nonparam --todos mesh2_window
            ;;
        debug)
            # srun -N 1 -n 4 -C "gpu&hbm80g" -t 00:30:00 --gpus 4 --qos debug --account desi_g python compute_mesh_jax.py --version AbacusHF-v2 --tracers QSO --domain altmtl --zerrs None --mockid 0 --regions NGC --todos mesh3_sugiyama_window --overwrite
            srun -N 1 -n 4 -C "gpu&hbm80g" -t 00:30:00 --gpus 4 --qos debug --account desi_g python compute_mesh_jax.py --version data-dr1-v1.5 --tracers QSO --domain altmtl --zranges "1.4-1.7" --ntile_cut 1 --todos mesh2
            ;;
    esac
}

# Check if a computation type was provided
if [ -z "$1" ]; then
    echo "Usage: bash srun.sh [cat|2pt|mesh|window]"
    exit 1
fi

# Run the srun command
activate_env $1
run_srun $1