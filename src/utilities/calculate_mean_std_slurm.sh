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

SIFFILE="/mnt/users/leobakh/VET_project/VET-Special-syllabus/singularity/container_pytorch_cv.sif" ## FILL INN

## RUN THE PYTHON SCRIPT
# Using a singularity container named container_u_net.sif
singularity exec --nv $SIFFILE python calculate_mean_std.py        ## FILL INN

# Send this job into the slurm queue with the following command: 
# >> sbatch test_script_slurm.sh 
