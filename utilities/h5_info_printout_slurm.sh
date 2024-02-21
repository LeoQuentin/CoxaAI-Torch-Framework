#!/bin/bash
#SBATCH --ntasks=16          # 16 cores (CPU)
#SBATCH --nodes=1            # Use 1 node
#SBATCH --job-name=training  # Name of job
#SBATCH --partition=gpu      # Use GPU partition
#SBATCH --gres=gpu:1         # Use one GPUs
#SBATCH --mem=64G            # Default memory per CPU is 3GB
#SBATCH --output=./output_logs/training_prints_%j.out # Stdout file   

## Script commands
module load singularity

SIFFILE="/mnt/users/leobakh/containerizing/container_pytorch_cv.sif" ## FILL INN

## RUN THE PYTHON SCRIPT
singularity exec --nv $SIFFILE python h5_info_printout.py

# Send this job into the slurm queue with the following command: 
# >> sbatch test_script_slurm.sh 