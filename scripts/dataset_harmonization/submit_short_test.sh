#! /bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeremiah.ridge@wolfson.ox.ac.uk
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --output=slurm_out/%j.out

echo "Starting job script"

source /data/engs-pnpl/wolf6942/miniforge3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo "Failed to source conda.sh"
    exit 1
fi

mamba activate megalodon
if [ $? -ne 0 ]; then
    echo "Failed to activate environment"
    exit 1
fi

srun python $@
if [ $? -ne 0 ]; then
    echo "Python script failed"
    exit 1
fi

echo "Job script completed successfully"
