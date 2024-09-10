#! /bin/bash

#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=subset_test
#SBATCH --time=49:59:00
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@

