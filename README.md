# MEGalodon
![A Megalodon](https://i.imgur.com/tvJLJ1X.png)


Let's learn powerful representations from MEG data.

## Contributing

### Prerequisites
Make sure you're on ARC so you have access to the datasets!

`conda create --name megalodon python=3.10.13`
`pip install -r requirements.txt`

### Repository structure
- `dataloaders/` Contains all dataloaders, split into dataloaders for self-supervised training and dataloaders for downstream training
- `models/` Contains all models for encoders, classifiers, etc.
- `scripts/` Slurm submission scripts for ARC cluster.

### Code standards
Be sensible (add comments where necessary) and use ruff for linting all files and organising imports. Try and decouple code where possible.