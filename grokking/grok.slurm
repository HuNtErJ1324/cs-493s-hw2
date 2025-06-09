#!/bin/bash

#SBATCH --job-name=hw2-grokking
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tlilly@uw.edu
#SBATCH --account=stf

#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=1-3

#SBATCH --time=15:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/gscratch/scrubbed/tlilly/cs-493s-hw2
#SBATCH --export=all
#SBATCH --output=grokking/grok_%x_%A_%a.out
#SBATCH --error=grokking/grok_%x_%A_%a.err

# Activate conda
source /gscratch/foster/sw/miniconda3/etc/profile.d/conda.sh
conda activate core

# Your programs to run.
srun python grok.py --run_opt $SLURM_ARRAY_TASK_ID
