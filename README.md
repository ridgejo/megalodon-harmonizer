# MEGalodon
<img src="https://i.imgur.com/tvJLJ1X.png" alt="Megalodon" width="400"/>


Let's learn powerful representations from MEG data.

## Contributing

### Prerequisites
- Make sure you're on ARC (so the code can access the datasets in the shared directory)
- Have installed miniconda (pro tip: install miniconda in $DATA and not $HOME)

```
# 1. Setup conda environment
conda create --name megalodon python=3.10.13
conda activate megalodon

# 2. Install OSL into environment
cd ~
git clone https://github.com/OHBA-analysis/osl.git
cd osl
pip install -e .

# 3. Clone MEGalodon and install requirements
cd ~
git clone git@github.com:neural-processing-lab/MEGalodon.git
cd MEGalodon
pip install -r requirements.txt
```

### Repository structure
- `dataloaders/` Contains all dataloaders, split into dataloaders for self-supervised training and dataloaders for downstream training
- `models/` Contains all models for encoders, classifiers, etc.
- `scripts/` Slurm submission scripts for ARC cluster.
- `configs/` Configuration files for training.

### Code standards
Use ruff for linting, formatting, and organising imports. I recommend running the following inside the repository before committing:
```
# Linting checks and automatic fixes
ruff check . --fix
# Formatting fixes
ruff check --select I --fix; ruff format .
```