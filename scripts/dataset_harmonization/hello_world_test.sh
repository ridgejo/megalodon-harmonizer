#!/bin/bash
#SBATCH --job-name=hello_slurm
#SBATCH --output=slurm_log/slurm-%j.out
#SBATCH --error=slurm_log/slurm-%j.err

# Basic test script
echo "Hello, SLURM!"

