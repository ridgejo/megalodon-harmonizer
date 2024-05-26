#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --qos=standard
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=168:59:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --output=slurm_out/%j.out

source ~/.init_conda.sh
conda activate pnpl_base

export WANDB_CACHE_DIR=$DATA/wandb_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun python $@