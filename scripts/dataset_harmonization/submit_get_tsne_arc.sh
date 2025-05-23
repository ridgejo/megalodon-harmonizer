#! /bin/bash

#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --qos=system
#SBATCH --clusters=arc
#SBATCH --job-name=getTSNE
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@

