#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --output=slurm_out/%j.out

source ~/.init_conda.sh
conda activate MEGalodon

export WANDB_CACHE_DIR=$DATA/wandb_cache

srun python $@
