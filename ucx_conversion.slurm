#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH -N 4
#SBATCH --partition=mi2104x
#SBATCH -n 4
#SBATCH --job-name=amd_ucx_conversion_mpi
#SBATCH --output=slurm_logs/%x_%j.out

IN_FOLDER="path/to/dcp_folder"
OUT_FOLDER="${IN_FOLDER}_universal_checkpoint"
N=8

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
srun --mpi=pmix python ds_to_universal_mpi_ucx.py --input_folder $IN_FOLDER --output_folder ${OUT_FOLDER} --num_workers $N --mpi