# MEGalodon
<img src="https://i.imgur.com/tvJLJ1X.png" alt="Megalodon" width="400"/>


Let's learn highly generalisable deep representations from MEG data.

## Contributing

### Prerequisites
- Make sure you're on ARC (so the code can access the datasets in the shared directory)
- Have installed miniconda (pro tip: install miniconda in $DATA and not $HOME)

```
# 1. Setup and activate conda environment
conda create --name megalodon python=3.10.13 && conda activate megalodon

# 2. Install our fork of OSL into the environment
git clone --single-branch --branch feature/fix_kit git@github.com:neural-processing-lab/osl.git
pip install -e ./osl

# 3. Install PNPL into environment
git clone git@github.com:neural-processing-lab/pnpl.git
pip install -e ./pnpl

# 4. Clone MEGalodon and install requirements
git clone git@github.com:neural-processing-lab/MEGalodon.git
cd MEGalodon
pip install -r requirements.txt
```

### Repository structure
- `dataloaders/` Contains all dataloaders, split into dataloaders for self-supervised training and dataloaders for downstream training
- `models/` Contains all models for encoders, classifiers, etc.
- `scripts/` Slurm submission scripts for ARC cluster.
- `configs/` Configuration files for training.

### Basic usage
Train a model:
`python train_rep.py --config configs/<> --name <>`

Or using one of the scripts:
`sbatch scripts/<>.sh train_rep.py --config configs/<> --name <>`

Checkpoints (based on minimum validation loss) will be saved to $DATA/experiments

Fine-tune or continue training a pre-trained model:

`python train_rep.py --config configs/<> --checkpoint <> --name <>`

If fine-tuning, make sure that the config specifies a `finetune` key as well as any downstream tasks to use for this. A good example is `configs/multi/armeni_voiced_encoder.yaml` which specifies SSL losses for pre-training and a downstream voiced classifier for fine-tuning.

### Code standards
Use ruff for linting, formatting, and organising imports. I recommend running the following inside the repository before committing:
```
# Linting checks and automatic fixes
ruff check . --fix
# Formatting fixes
ruff check --select I --fix; ruff format .
```
You could use a pre-commit hook for this instead by creating a `.git/hooks/pre-commit`:
```
#!/bin/bash

echo "Running ruff linting and formatting pre-commit hook"

ruff check . --fix

# Check command success
if [ $? -ne 0 ]; then
  echo "Pre-commit check failed."
  exit 1
fi

ruff check --select I --fix; ruff format .

# Check command success
if [ $? -ne 0 ]; then
  echo "Pre-commit check failed."
  exit 1
fi
```
and making it executable with `chmod +x .git/hooks/pre-commit` to automatically do these checks.