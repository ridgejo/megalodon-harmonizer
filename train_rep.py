import argparse
from pathlib import Path

import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataloaders.data_utils import DATA_PATH
from dataloaders.multi_dataloader import MultiDataLoader
from models.brain_encoders.rep_learner import RepLearner

parser = argparse.ArgumentParser(
    prog="MEGalodon-representation",
    description="Learn a representation from brain data.",
)
parser.add_argument("--config", help="Path to config file (yaml)", required=True)
parser.add_argument("--name", help="Name for run", default=None)
parser.add_argument(
    "--debug", help="Faster debug mode", action="store_true", default=False
)
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())

if args.debug:
    config["datamodule_config"]["debug"] = True

seed_everything(config["experiment"]["seed"], workers=True)

wandb_logger = WandbLogger(
    name=args.name,
    project=parser.prog,
    save_dir=DATA_PATH / "experiments",
    log_model="all",
    dir=DATA_PATH / "wandb",
)

wandb_logger.experiment.config.update(config)

# Checkpoint model only when validation loss improves
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    auto_insert_metric_name=True,
)

model = RepLearner(
    config["rep_config"],
    batch_size=config["datamodule_config"]["dataloader_configs"]["batch_size"],
)
datamodule = MultiDataLoader(**config["datamodule_config"])

# Track gradients
wandb_logger.watch(model)

trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=datamodule)
