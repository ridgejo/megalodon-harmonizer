#! /bin/bash

#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=dev_subset_test
#SBATCH --time=00:09:59
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@

