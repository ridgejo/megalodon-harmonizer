#! /bin/bash

#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=subset_test
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
#SBATCH --partition=long
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@

