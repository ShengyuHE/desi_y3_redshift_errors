#!/bin/bash
set -euo pipefail

# Extra run_fits.py arguments override the defaults below.
# Usage: bash srun_fit.sh [build|test|profile|sample] [arguments]
# Select the theory preset with FIT_THEORY (default: folpsEFT).
# FOLPS uses Chebyshev; build caches before profile/sample runs.

todo=sample #profile, sample
case "${1:-}" in
    build|test|profile|sample) todo=$1; shift ;;
esac

common_args=(
    --version AbacusHF-v2
    --domain altmtl
    --cov_version holi-v3
    --tracers LRG2
    --regions GCcomb
    --zerrs None
    --mockid 0-24
    --stats mesh2 mesh3
    --cosmo_params base
    --emulator-budget 3
    --sampler pocomc
)

theory_model=${FIT_THEORY:-folpsEFT} # folpsEFT, COMET
case "$theory_model" in
    folpsEFT)
        theory_args=(
            --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_error_fits/folpsEFT
            --theory_model folpsEFT
            --kmax 0.20-0.20-0.20-0.03
        )
        ;;
    COMET)
        # Shared model tables installed with the cosmodesi environment.
        export COMET_DATA_DIR=${COMET_DATA_DIR:-/global/cfs/cdirs/desi/science/cpe/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/desilike/comet-emulators}
        # Scale cuts for P0, P2, B000, and B202, respectively.
        theory_args=(
            --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_error_fits/COMET
            --theory_model COMET
            --comet-model VDG_infty
            --kmax 0.35-0.30-0.20-0.10
        )
        ;;
    *) echo "Unknown FIT_THEORY: $theory_model" >&2; exit 2 ;;
esac

resources=(-n 4 -c 32 --gpu-bind=single:1)
threads=32
if [[ $todo == build ]]; then
    # Match example_emulator.py: CPU evaluations distributed over 16 ranks.
    resources=(-n 16 -c 8)
    threads=4
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command=(srun -N 1 "${resources[@]}" -C gpu --gpus-per-node=4
    --exclusive -t "${FIT_TIME:-04:00:00}" --qos "${FIT_QOS:-interactive}" --account desi_g
    python -u "$script_dir/run_fits.py" "${common_args[@]}" "${theory_args[@]}" --todos "$todo" "$@")

source /global/homes/s/shengyu/env.sh fit_env
if [[ $todo == build ]]; then
    export JAX_PLATFORMS=cpu
    export MPICH_GPU_SUPPORT_ENABLED=0
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$threads}
export PYTHONDONTWRITEBYTECODE=1
"${command[@]}"