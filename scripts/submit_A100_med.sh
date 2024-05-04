#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=47:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=medium
#SBATCH --output=slurm_out/%j.out

source ~/.init_conda.sh
conda activate MEGalodon

export WANDB_CACHE_DIR=$DATA/wandb_cache

python $@
