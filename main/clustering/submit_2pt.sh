#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH -t 22:00:00
#SBATCH -C gpu
#SBATCH --gpus=4
#SBATCH -q regular
#SBATCH -A desi_g
#SBATCH --array=0-2
#SBATCH --output=./slurms/2pt/cutsky-id-%A_%a.out

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

MZRR=("False" "global" "bin")
ZRR="${MZRR[$SLURM_ARRAY_TASK_ID]}"
echo "Node $(hostname) running ZRR=$ZRR"
python compute_2pt.py --domain cutsky --tracer ELG --mockid 0-24 --mzrr $ZRR --corr pk
echo "Finished $ZRR"

'''
# SBATCH --array=0-2
Redshift-error modes: no zerr / global / binned
MZRR=(False global bin)
ZRR="${MZRR[$SLURM_ARRAY_TASK_ID]}"
echo "Node $(hostname) running ZRR=$ZRR"
''