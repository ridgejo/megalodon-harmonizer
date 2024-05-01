import argparse
import glob
import os
from pathlib import Path

import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner import Tuner

from dataloaders.data_utils import DATA_PATH
from dataloaders.data_module import MEGDataModule
from models.brain_encoders.rep_learner import RepLearner

# Increase wandb waiting time to avoid timeouts
os.environ["WANDB__SERVICE_WAIT"] = "300"

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
    "--anomaly_detect", help="Detect anomalies", action="store_true", default=False
)
parser.add_argument(
    "--lr_find", help="Find best learning rate", action="store_true", default=False
)
parser.add_argument("--ddp", help="Use DDP", action="store_true", default=False)
parser.add_argument(
    "--profile", help="Use profiling", action="store_true", default=False
)
parser.add_argument("--seed", help="Override experiment seed", type=int, default=None)
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())

if args.debug:
    config["datamodule_config"]["debug"] = True

if args.seed is not None:
    config["experiment"]["seed"] = args.seed

seed_everything(config["experiment"]["seed"], workers=True)

wandb_logger = WandbLogger(
    name=args.name,
    project=parser.prog,
    save_dir=DATA_PATH / "experiments",
    log_model=True,  # Log checkpoint only at the end of training (to stop my wandb running out of storage!)
    dir=DATA_PATH / "wandb",
)

wandb_logger.experiment.config.update(config)

# Checkpoint model only when validation loss improves
val_checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    auto_insert_metric_name=True,
)

latest_checkpoint = ModelCheckpoint(
    filename="latest-checkpoint",
    every_n_epochs=1,
    save_top_k=1,
)

datamodule = MEGDataModule(
    **config["datamodule_config"],
    seed=config["experiment"]["seed"],
)

ddp_strategy = DDPStrategy(find_unused_parameters=True, static_graph=True)

# How to handle checkpoints? If a checkpoint is specified, then let the config be a special fine-tuning config which
# only specifies data and fine-tuning parameters. Everything else can be loaded directly from the checkpoint.
resume_training = False
if args.checkpoint:
    # Let's support a couple of *different* modes if a checkpoint is specified:
    # a) Resume training
    # b) Fine-tuning

    # If not fine-tuning, we can just continue from the checkpoint
    if "finetune" in config:
        # Fine-tuning case: use a special fine-tuning config for this.

        # Get checkpoint file
        if args.checkpoint[-5:] == ".ckpt":
            checkpoint = args.checkpoint
        elif args.checkpoint == "random":
            checkpoint = "random"
        else:
            checkpoint = glob.glob(args.checkpoint + "/**/epoch*.ckpt")[
                0
            ]  # Find first validation checkpoint file within the directory

        # Load model from the pre-trained checkpoint
        if checkpoint != "random":
            model = RepLearner.load_from_checkpoint(
                checkpoint, rep_config=config["rep_config"]
            )
        else:
            # Use random model initialisation (e.g. for baseline)
            model = RepLearner(
                config["rep_config"],
            )

        if config["finetune"]["freeze_all"]:
            # Freeze all layers except any downstream classifiers that are already enabled
            model.freeze_except("classifier")
            # Remove other losses / predictors from the model
            model.disable_ssl()
            # warning: also removes any existing classifiers from pre-training stage
            model.disable_classifiers()
        elif "new_subject" in config["finetune"]:
            model.freeze_except("subject_")  # Leave any subject conditioning unfrozen
            # Do not disable SSL
            model.disable_classifiers()
        else:
            model.disable_ssl()
            model.disable_classifiers()

        # Add new downstream classifiers to the model
        for k, v in config["finetune"].items():
            if "classifier" in k:
                model.add_classifier(k, v)

        # Make sure configure_optimizer is called *after* this
        model.configure_optimizers()

        # There can be unused parameters, but these remain fixed throughout training
        ddp_strategy = DDPStrategy(find_unused_parameters=True, static_graph=True)

    else:
        resume_training = True

        # Get checkpoint file
        if args.checkpoint[-5:] == ".ckpt":
            checkpoint = args.checkpoint
        else:
            checkpoint = glob.glob(args.checkpoint + "/**/latest*.ckpt")[
                0
            ]  # Find latest checkpoint file within the directory

        # Load model from the pre-trained checkpoint and resume training
        model = RepLearner.load_from_checkpoint(
            checkpoint, rep_config=config["rep_config"]
        )

else:
    model = RepLearner(
        config["rep_config"],
    )

# Track gradients
wandb_logger.watch(model)

epochs = config["experiment"]["epochs"] if "epochs" in config["experiment"] else 1000
epochs = 10 if args.profile else epochs

trainer = Trainer(
    logger=wandb_logger,
    callbacks=[latest_checkpoint, val_checkpoint],
    detect_anomaly=args.anomaly_detect,
    strategy="auto" if not args.ddp else ddp_strategy,
    max_epochs=epochs,
    profiler="simple" if args.profile else None,
)

if args.lr_find:
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    print("Learning rate search results:")
    print(lr_finder.results)

if resume_training:
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
else:
    trainer.fit(model, datamodule=datamodule)

# Automatically tests model with best weights from training/fitting
print("Testing model")
trainer.test(datamodule=datamodule)
