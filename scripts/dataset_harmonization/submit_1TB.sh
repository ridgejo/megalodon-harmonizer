#! /bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeremiah.ridge@wolfson.ox.ac.uk
#SBATCH --nodes=1
#SBATCH --mem=1200G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=ptMEGall
#SBATCH --time=175:00:00
#SBATCH --gres=gpu:1 --constraint='gpu_sku:A100'
#SBATCH --partition=test
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@
