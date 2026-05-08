#!/bin/bash
#SBATCH -N 140
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -A desi_g
#SBATCH --chdir=/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering
#SBATCH --output=/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering/slurm_logs/ELG_%j.out

set -euo pipefail
# source /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250526-1.0.0/conda/etc/profile.d/conda.sh
# source /global/homes/s/shengyu/env.sh 2pt_env
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MPIIO_DVS_MAXNODES=1

MAX_PARALLEL=${SLURM_JOB_NUM_NODES:-20}

mapfile -t JOB_SPECS < <(python - "${MAX_PARALLEL}" <<'PY'
import sys
from pathlib import Path

max_parallel = int(sys.argv[1])
base = Path('/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/holi-v3/altmtl/ELG')
skip = {363, 565}
expected = [mock_id for mock_id in range(1000) if mock_id not in skip]

filenames = {
    'None': [
        'mesh2_spectrum_poles_ELG_z0.8-1.1_holi_v3.h5',
        'mesh2_spectrum_poles_ELG_z1.1-1.6_holi_v3.h5',
    ],
    'verr_nonparam': [
        'mesh2_spectrum_poles_ELG_z0.8-1.1_holi_v3+dv_verr_nonparam.h5',
        'mesh2_spectrum_poles_ELG_z1.1-1.6_holi_v3+dv_verr_nonparam.h5',
    ],
}

missing_by_zerr = {}
for zerr, names in filenames.items():
    missing = []
    for mock_id in expected:
        mock_dir = base / f'mock{mock_id}' / 'mpspk'
        if any(not (mock_dir / name).exists() for name in names):
            missing.append(mock_id)

    print(f'# {zerr}: {len(missing)} mocks missing at least one ELG bin', file=sys.stderr)
    missing_by_zerr[zerr] = missing

for inode in range(max_parallel):
    none_ids = missing_by_zerr['None'][inode::max_parallel]
    verr_ids = missing_by_zerr['verr_nonparam'][inode::max_parallel]
    if none_ids or verr_ids:
        none_label = ','.join(map(str, none_ids)) if none_ids else '-'
        verr_label = ','.join(map(str, verr_ids)) if verr_ids else '-'
        print(f'{none_label}:{verr_label}')
PY
)

if (( ${#JOB_SPECS[@]} == 0 )); then
    echo "All ELG mesh2 measurements are already complete."
    exit 0
fi

echo "Launching ${#JOB_SPECS[@]} ELG completion chunks on up to ${MAX_PARALLEL} nodes."

running=0
status=0
for task_id in "${!JOB_SPECS[@]}"; do
    IFS=: read -r NONE_IDS VERR_IDS <<< "${JOB_SPECS[$task_id]}"

    echo "Launching task ${task_id}: ELG None=${NONE_IDS} verr_nonparam=${VERR_IDS}"
    srun --exclusive -N 1 -n 4 --gpus-per-node=4 bash -c '
        set -euo pipefail
        if [[ "$1" != "-" ]]; then
            python compute_mesh_jax.py --version holi-v3 --domain altmtl --tracers ELG --mockid "$1" --zerrs None --todos mesh2
        fi
        if [[ "$2" != "-" ]]; then
            python compute_mesh_jax.py --version holi-v3 --domain altmtl --tracers ELG --mockid "$2" --zerrs verr_nonparam --todos mesh2
        fi
    ' _ "${NONE_IDS}" "${VERR_IDS}" &

    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
        if ! wait -n; then
            status=1
        fi
        running=$((running - 1))
    fi
done

if ! wait; then
    status=1
fi
exit "${status}"
