#! /bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeremiah.ridge@wolfson.ox.ac.uk
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=subset_test
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@

