#!/bin/bash
#SBATCH --job-name=exampleJob
#SBATCH --account=project_465000861
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=small-g

# Load modules and reload git-lfs
module --force purge
export EBU_USER_PREFIX=/project/project_465000861/EasyBuild
module load LUMI/23.09
module load EasyBuild-user
clear-eb
eb git-lfs.3.3.0.eb -r
module load LUMI/23.09
module load partition/L
module load git-lfs/3.3.0

# Set max_split_size_mb for memory management
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:500

# Print working directory
pwd

# Activating env, logging in to huggingface and running script
srun singularity exec --bind /scratch/project_465000861:/scratch/project_465000861 --bind /projappl/project_465000861/EasyBuild/SW/LUMI-23.09/L/git-lfs/3.3.0/bin/git-lfs:/users/schleder/research_env/bin/git-lfs /scratch/project_465000861/pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif /bin/bash -c "source ~/research_env/bin/activate; huggingface-cli login --token '<TOKEN>'; python mainDANSUM_lumi.py"