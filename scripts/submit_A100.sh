#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=short
#SBATCH --output=slurm_out/%j.out

source ~/.init_conda.sh
conda activate pnpl_base

export WANDB_CACHE_DIR=$DATA/wandb_cache

python $@
