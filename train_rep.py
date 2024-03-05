import argparse
import yaml

from pathlib import Path
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

from models.brain_encoders.rep_learner import RepLearner
from dataloaders.multi_dataloader import MultiDataLoader
from dataloaders.data_utils import DATA_PATH

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

seed_everything(config["experiment"]["seed"], workers=True)

wandb_logger = WandbLogger(
    project=parser.prog,
    save_dir=DATA_PATH / "experiments",
    log_model="all",
    dir=DATA_PATH / "wandb",
)

model = RepLearner(
    config["rep_config"],
    batch_size=config["datamodule_config"]["dataloader_configs"]["batch_size"]
)
datamodule = MultiDataLoader(**config["datamodule_config"])

trainer = Trainer(logger=wandb_logger)
trainer.fit(model, datamodule=datamodule)