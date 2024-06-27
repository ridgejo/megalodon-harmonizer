#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=test
#SBATCH --output=slurm_out/%j.out

source ~/.init_conda.sh
conda activate pnpl_base

export WANDB_CACHE_DIR=$DATA/wandb_cache

srun python $@
