import argparse
from pathlib import Path

import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

from dataloaders.data_utils import DATA_PATH
from dataloaders.multi_dataloader import MultiDataLoader
from models.brain_encoders.rep_learner import RepLearner

parser = argparse.ArgumentParser(
    prog="MEGalodon-representation",
    description="Learn a representation from brain data.",
)
parser.add_argument("--config", help="Path to config file (yaml)", required=True)
parser.add_argument(
    "--checkpoint", help="Path to existing weights to load representation", default=None
)
parser.add_argument("--name", help="Name for run", default=None)
parser.add_argument(
    "--debug", help="Faster debug mode", action="store_true", default=False
)
parser.add_argument(
    "--lr_find", help="Find best learning rate", action="store_true", default=False
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

if args.checkpoint:
    model = RepLearner.load_from_checkpoint(args.checkpoint)
    # model.freeze_except()
    # todo: make sure configure_optimizer is called *after* this
    # model.configure_optimizer() ???
else:
    model = RepLearner(
        config["rep_config"],
    )
datamodule = MultiDataLoader(**config["datamodule_config"])

# Track gradients
wandb_logger.watch(model)

trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])

if args.lr_find:
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    print("Learning rate search results:")
    print(lr_finder.results)

trainer.fit(model, datamodule=datamodule)
