#!/bin/bash
#SBATCH --ntasks-per-node=32          # 16 cores (CPU)
#SBATCH --nodes=1            # Use 1 node
#SBATCH --job-name=CoxaAI-ViT  # Name of job
#SBATCH --partition=gpu      # Use GPU partition
#SBATCH --gres=gpu:1         # Use one GPUs
#SBATCH --mem=64G            # Default memory per CPU is 3GB
#SBATCH --output=./output_logs/training_prints_%j.out # Stdout file

## Script commands
module load singularity

SIFFILE="/mnt/users/leobakh/VET_project/VET-Special-syllabus/singularity/container_CoxaAI_Poetry.sif" ## FILL INN
## RUN THE PYTHON SCRIPT
# Using a singularity container named container_u_net.sif
singularity exec --nv $SIFFILE python Swin_lightaugreg.py
# Send this job into the slurm queue with the following command:
# >> sbatch test_script_slurm.sh
