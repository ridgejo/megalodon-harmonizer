#! /bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeremiah.ridge@wolfson.ox.ac.uk
#SBATCH --nodes=1
#SBATCH --qos=system
#SBATCH --job-name=devel_megalodon
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --output=slurm_log/slurm-%j.out
#SBATCH --error=slurm_log/slurm-%j.err

# source env
source scripts/dataset_harmonization/source_megalodon_env.sh

# run exp
srun python $@
