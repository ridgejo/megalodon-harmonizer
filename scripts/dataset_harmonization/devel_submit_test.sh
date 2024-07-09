#! /bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeremiah.ridge@wolfson.ox.ac.uk
#SBATCH --nodes=1
#SBATCH --qos=system
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=devel_megalodon
#SBATCH --time=00:09:59
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --output=slurm_log/slurm-%j.out

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@
