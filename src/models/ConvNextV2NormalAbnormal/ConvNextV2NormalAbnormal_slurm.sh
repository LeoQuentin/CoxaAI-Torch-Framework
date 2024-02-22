#!/bin/bash

# Define the Python script filename
filename="ConvNextV2NormalAbnormal.py"

#SBATCH --ntasks-per-node=16 # 16 cores (CPU)
#SBATCH --nodes=1            # Use 1 node
#SBATCH --job-name=training  # Name of job
#SBATCH --partition=gpu      # Use GPU partition
#SBATCH --gres=gpu:1         # Use one GPUs
#SBATCH --mem=64G            # Default memory per CPU is 3GB
#SBATCH --output=/mnt/users/leobakh/VET_project/VET-Special-syllabus/logs/output_logs/${filename}_%j.out  # Dynamic output filename based on Python script name
## Script commands
module load singularity

SIFFILE="/mnt/users/leobakh/VET_project/VET-Special-syllabus/singularity/containers/container_pytorch_cv_with_metrics.sif" ## FILL INN

## RUN THE PYTHON SCRIPT
singularity exec --nv $SIFFILE python $filename

# Send this job into the slurm queue with the following command: 
# >> sbatch test_script_slurm.sh 