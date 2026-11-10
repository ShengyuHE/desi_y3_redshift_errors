#!/bin/bash
set -euo pipefail

# QSO-only fits using the resource and theory presets in srun_fit.sh.
# Usage: bash srun_QSO_fit.sh [build|test|profile|sample] [--corr_dv MODE] [arguments]
# Set fit_theory below; FIT_THEORY can override it from the environment.
# CORR_DV and REDSHIFT_ERROR_PDF set defaults; command-line options override them.
# Theory correction is off by default; --corr_dv None explicitly disables it.

fit_theory=${FIT_THEORY:-COMET} # folpsEFT or COMET

todo=sample
case "${1:-}" in
    build|test|profile|sample) todo=$1; shift ;;
esac

corr_dv=${CORR_DV:-None}
pdf_file=${REDSHIFT_ERROR_PDF:-/pscratch/sd/s/shengyu/repeats/DA2/loa-v1/verr_mode/CDF_verr_nonparam_QSO_z0.8-2.1.npz}
extra_args=()
while (( $# )); do
    case "$1" in
        --corr_dv|--redshift-error-pdf)
            if (( $# < 2 )) || [[ $2 == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ $1 == --corr_dv ]]; then corr_dv=$2; else pdf_file=$2; fi
            shift 2
            ;;
        --corr_dv=*) corr_dv=${1#*=}; shift ;;
        --redshift-error-pdf=*) pdf_file=${1#*=}; shift ;;
        *) extra_args+=("$1"); shift ;;
    esac
done

correction_args=(--corr_dv "$corr_dv")
case "$corr_dv" in
    None|none) correction_args=() ;;
    pdf|pdf+gauss|pdf+lor)
        correction_args+=(--redshift-error-pdf "$pdf_file")
        ;;
    gauss|lor) ;;
    *) echo "Unknown corr_dv mode: $corr_dv (choose None, gauss, lor, pdf, pdf+gauss, pdf+lor)" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FIT_THEORY="$fit_theory"
exec bash "$script_dir/srun_fit.sh" "$todo" \
    --tracers QSO1 --stats mesh2 --zerrs None verr_nonparam "${correction_args[@]}" "${extra_args[@]}"