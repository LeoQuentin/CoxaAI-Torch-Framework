# Exploring State-of-the-Art Models in Veterinary Imaging
This project is an exploration into the application of modern neural network architectures for the classification of hip dysplasia in dog X-rays. It builds on the project [CoxaAI](https://github.com/huynhngoc/coxaai/tree/master). Using private data from the CoxaAI project, this effort aims to experiment with the latest advancements in deep learning. 
Hugging Face models and PyTorch Lightning are at the center of this project.

## Getting Started
This project is built specifically for use on data that is not publically available, on a private SLURM-compute cluster. Therefore, adapting this project for outside use will take some work. If you for some reason are sitting on a few thousand 800x800 medical grayscale images and a SLURM-server with GPUs, setting this project is as simple as cloning it, changing the filepaths everywhere and editing the `dataset.py` file to load your data according to your setup. 
If you have all of the above, training a model is as simple as running `sbatch {model_name}_slurm.sh`. Training logs will automatically be saved, the model from the best epoch will be saved, and so on. You may have to change the `dataset.py` file when changing datasets, as the handling of the h5 file is very specific to our current dataset `hips_800_sort_4.h5`.

## Notes about Project Structure:
logs: Contains all logs related to the project.
	- container_logs: Stores logs from the creation of the Singularity containers.
	- loss_logs: Contains logs related to model training losses and performance metrics, organized by model and experiment version.
	- output_logs: Contains output files from training sessions.

singularity: 
	- Includes Singularity definition files (*.def) and scripts for creating and managing Singularity containers. Note: The building of a singularity needs to be run on a machine with a GPU, otherwise there will be Nvidia driver issues.

models:
	- Contains directories for each model being experimented with, including Python scripts for the models and SLURM scripts for job submission. 
	- See guidelines below for naming conventions

utilities:
	- Holds utility scripts for tasks such as data preprocessing, dataset analysis, and other supportive functions.

trained_models: 
	- Stores the best-performing model checkpoints for use in further evaluation or application.
	- Stored privately and not put on github.

## TODO:
	- Implement and test more models
	- Create scripts/notebooks to visualize performance
	- Avoid hardcoding paths for everything (by using a .env file or something?)