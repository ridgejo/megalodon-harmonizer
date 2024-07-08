#!/bin/bash

# Load the required conda script to use mamba
source /data/engs-pnpl/wolf6942/miniforge3/etc/profile.d/conda.sh

# Get the relative path to the bash directory
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

# Activate the mamba environment
conda activate megalodon
