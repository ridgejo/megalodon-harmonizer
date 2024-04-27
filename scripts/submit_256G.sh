#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=11:30:00
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
#SBATCH --partition=short
#SBATCH --output=slurm_out/%j.out

source ~/.init_conda.sh
conda activate MEGalodon

export WANDB_CACHE_DIR=$DATA/wandb_cache

python $@
