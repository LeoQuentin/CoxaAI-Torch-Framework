#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=CoxaAI-ViT  # Name of job
#SBATCH --partition=gpu      # Use GPU partition
#SBATCH --gpus-per-node=2  # Request 2 GPUs per node 
#SBATCH --mem=64G            # Default memory per CPU is 3GB
#SBATCH --output=./output_logs/training_prints_%j.out # Stdout file

## Script commands
module load singularity

SIFFILE="/mnt/users/leobakh/VET_project/VET-Special-syllabus/singularity/container_CoxaAI_Poetry.sif" ## FILL INN
## RUN THE PYTHON SCRIPT
# Using a singularity container named container_u_net.sif
srun singularity exec --nv $SIFFILE python SwinBinaryVsMulticlass.py
# Send this job into the slurm queue with the following command:
# >> sbatch test_script_slurm.sh
