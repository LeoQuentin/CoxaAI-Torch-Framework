#!/bin/bash
#SBATCH --ntasks=1			                # 1 core (CPU)
#SBATCH --nodes=1			                # Use 1 node
#SBATCH --job-name=build_singularity        # Name of job
#SBATCH --mem=3G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./log_create_container.out # Stdout and stderr file


## Script commands
# This is a script to generate new singularity container .sif file taking a .def file as an argument. 
# The script will use the same name for the .sif file it returns as the .def file has (hello.def --> hello.sif)
# The script works by submitting singularity creation as a SLURM job. 
# This has to be done because creating a GPU capable singularity requires the use of a GPU node, and 
# using interactive login commands in a shell script does not work. 
# To run script: 
# >> sbatch slurm_create_singularity_container.sh basepath/SINGULARITY_DEFINITION_FILE.def
# returns: 
#   * basepath/SINGULARITY_DEFINITION_FILE.sif     => singularity container
#   * basepath/log_SINGULARITY_DEFINITION_FILE.out => log file containing print statements from generation process (should be checked)

CONTAINERS_DIR = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/singularity/containers"
LOGS_DIR = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/logs/container_logs"

if [ $# = 0 ]
then
    echo "Error. No definition file given"
elif [ $# -gt 1 ]
then
    echo "Too many arguments given. Do one file at the time"
else
    BASENAME=$(basename "$1")
    SIF_FILENAME="container_${BASENAME%.*}.sif"
    SIF_PATH="${CONTAINERS_DIR}/${SIF_FILENAME}"
    LOG_FILENAME="log_create_container_${BASENAME%.*}.out"
    LOG_PATH="${LOGS_DIR}/${LOG_FILENAME}"

    echo "Making container $SIF_FILENAME from file $BASENAME, and putting it in $CONTAINERS_DIR"

    module load singularity
    singularity build --fakeroot $SIF_PATH $1

    # Move the log file to the correct directory
    # Since the script is executed inside definitions_scripts_logs, no need to move the log file.
    # Updated to reflect the correct log file handling
    echo "Log file located at: $LOG_PATH"
fi
