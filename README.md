# MEGalodon
<img src="https://i.imgur.com/tvJLJ1X.png" alt="Megalodon" width="400"/>


Let's learn powerful representations from MEG data.

## Contributing

### Prerequisites
- Make sure you're on ARC (so the code can access the datasets in the shared directory)
- Have installed miniconda

```
conda create --name megalodon python=3.10.13
pip install -r requirements.txt
conda activate megalodon
```

### Repository structure
- `dataloaders/` Contains all dataloaders, split into dataloaders for self-supervised training and dataloaders for downstream training
- `models/` Contains all models for encoders, classifiers, etc.
- `scripts/` Slurm submission scripts for ARC cluster.
- `configs/` Configuration files for training.

### Code standards
Use ruff for linting, formatting, and organising imports.