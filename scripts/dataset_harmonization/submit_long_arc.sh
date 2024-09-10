#! /bin/bash

#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --qos=system
#SBATCH --clusters=arc
#SBATCH --job-name=subset_test
#SBATCH --time=29:59:00
#SBATCH --partition=long
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@

