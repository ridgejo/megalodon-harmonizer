import argparse
import glob
import os
from pathlib import Path
import torch

import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner import Tuner

from dataloaders.data_module import HarmonizationDataModule, MEGDataModule
from dataloaders.data_utils import DATA_PATH
from models.brain_encoders.rep_harmonizer import RepHarmonizer

# Increase wandb waiting time to avoid timeouts
os.environ["WANDB__SERVICE_WAIT"] = "300"

SAVE_PATH = Path("/data/engs-pnpl/wolf6942")

parser = argparse.ArgumentParser(
    prog="MEGalodon-rep-harmonization",
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
parser.add_argument("--early_stop", help="Use early stopping checkpoint", action="store_true", default=False)
parser.add_argument("--full_run", help="Training on full datasets", action="store_true", default=False)
parser.add_argument("--get_tsne", help="Get TSNE plots for final encoder layer", action="store_true", default=False)
parser.add_argument("--sdat", help="Use SDAT optimization framework for unlearning", action="store_true", default=False)
parser.add_argument("--sgd", help="Use SGD for domain classifier during unlearning", action="store_true", default=False)
parser.add_argument("--clear_optim", help="Clear optimizer state upon loading checkpoint", action="store_true", default=False)
parser.add_argument("--clear_betas", help="Clear optimizer state betas, weight decay upon loading checkpoint", action="store_true", default=False)
parser.add_argument("--dset_num", help="Override num dataset domains", type=int, default=None)
parser.add_argument("--intersect_only", help="Only harmonize on intersect of distributions", action="store_true", default=False)
parser.add_argument("--lsvm", help="use LSVM for domain classifier", action="store_true", default=False)
parser.add_argument("--harmonization_lr", help="Override lr for harmonization stage", type=float, default=None)
parser.add_argument("--epoch_stage_1", help="Epoch to begin harmonization at", type=int, default=None)
parser.add_argument("--batch_size", help="Override batch side", type=int, default=None)
parser.add_argument("--no_dm_control", help="Don't include domain classifier in step 1 optimizer", action="store_true", default=False)
parser.add_argument("--multi_dm_pred", help="Use one classifier per task representation", action="store_true", default=False)
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())

# share experiment config val with lightning module
config["rep_config"]["max_epochs"] = config["experiment"]["epochs"]

# check fine tuning
if "finetune" in config:
    config["rep_config"]["epoch_stage_1"] = config["experiment"]["epochs"] + 1
    config["rep_config"]["finetune"] = True

if args.debug:
    config["datamodule_config"]["debug"] = True

if args.batch_size is not None:
    config["datamodule_config"]["dataloader_configs"]["batch_size"] = args.batch_size
config["rep_config"]["batch_size"] = config["datamodule_config"]["dataloader_configs"]["batch_size"]


if args.seed is not None:
    config["experiment"]["seed"] = args.seed

if args.name is not None:
    config["rep_config"]["run_name"] = args.name

if args.get_tsne:
    config["rep_config"]["tsne"] = True

if args.multi_dm_pred:
    config["rep_config"]["multi_dm_pred"] = True

if args.intersect_only:
    config["rep_config"]["intersect_only"] = True

if args.no_dm_control:
    config["rep_config"]["no_dm_control"] = True

if args.sdat:
    config["rep_config"]["sdat"] = True

if args.sgd:
    config["rep_config"]["sgd"] = True

if args.lsvm:
    config["rep_config"]["lsvm"] = True

if args.clear_optim:
    config["rep_config"]["clear_optim"] = True

if args.clear_betas:
    config["rep_config"]["clear_betas"] = True

if args.epoch_stage_1 is not None:
    config["rep_config"]["epoch_stage_1"] = args.epoch_stage_1

if args.harmonization_lr is not None:
    config["rep_config"]["dm_lr"] = args.harmonization_lr
    config["rep_config"]["conf_lr"] = args.harmonization_lr
    config["rep_config"]["task_lr"] = args.harmonization_lr

seed_everything(config["experiment"]["seed"], workers=True)

if args.full_run:
    exp_path = SAVE_PATH / "experiments" / "MEGalodon" / "full_run"
    config["rep_config"]["full_run"] = True
else:
    exp_path = SAVE_PATH / "experiments" / "MEGalodon"

wandb_logger = WandbLogger(
    name=args.name,
    project=parser.prog,
    save_dir=exp_path,
    log_model=True,  # Log checkpoint only at the end of training (to stop my wandb running out of storage!)
    dir=SAVE_PATH / "wandb" / "MEGalodon",
)

try:
    wandb_logger.experiment.config.update(config)
except Exception as _:
    print("Skipping rank > 0 wandb logging")

# Checkpoint model only when validation loss improves
val_checkpoint = ModelCheckpoint(
    dirpath = exp_path / "MEGalodon-rep-harmonization" / args.name, 
    monitor="val_loss",
    mode="min",
    auto_insert_metric_name=True,
)

latest_checkpoint = ModelCheckpoint(
    dirpath = exp_path / "MEGalodon-rep-harmonization" / args.name,
    filename="latest-checkpoint",
    every_n_epochs=1,
    save_top_k=1,
)

unlearning_checkpoint = ModelCheckpoint(
    dirpath = exp_path / "MEGalodon-rep-harmonization" / args.name,
    filename="{epoch}-UL-checkpoint",
    # every_n_epochs=config['rep_config']['epoch_stage_1'] - 1, #TODO val doesn't exists for FT conf
    every_n_epochs=config["rep_config"]["epoch_stage_1"],
    save_top_k=1,
    # enable_version_counter=True
)

if args.early_stop:
    early_stopping = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        patience=config['rep_config']['patience'],          # number of epochs with no improvement after which training will be stopped
        mode='min'           # mode can be 'min' or 'max'
    )

# # Custom callback to save checkpoint halfway through training
# class HalfwayCheckpoint(Callback):
#     def on_epoch_end(self, trainer, pl_module):
#         if trainer.current_epoch == (trainer.max_epochs / 2) - 1:
#             # Save checkpoint
#             print("Saving Halfway Checkpoint...")
#             checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, f"epoch_{trainer.current_epoch}.ckpt")
#             trainer.save_checkpoint(checkpoint_path)
#             print(f"Checkpoint saved at {checkpoint_path}")

# halfway_checkpoint = HalfwayCheckpoint()

if "finetune" in config:
    datamodule = MEGDataModule(        
        **config["datamodule_config"],
        seed=config["experiment"]["seed"],
    )
else:
    datamodule = HarmonizationDataModule(
        **config["datamodule_config"],
        seed=config["experiment"]["seed"],
    )
    if args.dset_num is not None:
        config["rep_config"]["num_datasets"] = args.dset_num
    else:
        dset_names = config["datamodule_config"]["dataset_preproc_configs"].keys()
        if "shaftoIntersection" in dset_names:
            num_dsets = len(dset_names) - 1
        else:
            num_dsets = len(dset_names)
        config["rep_config"]["num_datasets"] = num_dsets

ddp_strategy = DDPStrategy(
    find_unused_parameters=True, static_graph=False
)  # find_unused_parameters is not necessary here

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
            model = RepHarmonizer.load_from_checkpoint(
                checkpoint, rep_config=config["rep_config"]
            )
        else:
            # Use random model initialisation (e.g. for baseline)
            model = RepHarmonizer(
                config["rep_config"],
            )

        if config["finetune"]["freeze_all"]:
            model.finetuning_mode()
            # # Freeze all layers except any downstream classifiers that are already enabled
            # model.freeze_except("classifier")
            # # Remove other losses / predictors from the model
            # model.disable_ssl()
            # # warning: also removes any existing classifiers from pre-training stage
            # model.disable_classifiers()
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

        # Load the checkpoint to log epoch
        if args.get_tsne:
            print(f"Checkpoint loading model saved at epoch{torch.load(checkpoint, map_location=torch.device('cpu'))['epoch']}", flush=True)
        else:
            print(f"Checkpoint loading model saved at epoch{torch.load(checkpoint)['epoch']}", flush=True)

        # Load model from the pre-trained checkpoint and resume training
        model = RepHarmonizer.load_from_checkpoint(
            checkpoint, rep_config=config["rep_config"]
        )

else:
    model = RepHarmonizer(
        config["rep_config"],
    )

# Track gradients
wandb_logger.watch(model)

epochs = config["experiment"]["epochs"] if "epochs" in config["experiment"] else 1000
epochs = 10 if args.profile else epochs

if args.early_stop:
    callbacks = [latest_checkpoint, val_checkpoint, unlearning_checkpoint, early_stopping]
elif "finetune" in config:
    callbacks = [latest_checkpoint, val_checkpoint]
else:
    callbacks = [latest_checkpoint, unlearning_checkpoint, val_checkpoint]

trainer = Trainer(
    logger=wandb_logger,
    callbacks=callbacks,
    detect_anomaly=args.anomaly_detect,
    strategy="auto" if not args.ddp else ddp_strategy,
    max_epochs=epochs,
    profiler="simple" if args.profile else None,
    devices=4 if args.ddp else 1,
    default_root_dir= exp_path,
    num_sanity_val_steps=0
)

if args.lr_find:
    tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, datamodule=datamodule)
    lr_finder = tuner.lr_find(model, datamodule=datamodule, attr_name="dm_learning_rate")
    print("Learning rate search results:")
    print(lr_finder.results)

if args.get_tsne:
    # Get one batch from the validation dataloader
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()
    batch = next(iter(test_dataloader))

    device = "cpu"
    # Move model to CPU
    model.to(device)

    # Ensure batch on CPU by iterating over the tensors
    for b in batch:
        b['data'] = b['data'].float().to(device)

    # Call the validation step
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # T-SNE plot made and saved in val step
        model.get_tsne(batch=batch, name=args.name)
else:
    if resume_training:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Automatically tests model with best weights from training/fitting
    print("Testing model")

    if "test_datamodule_config" in config:
        del datamodule
        if "finetune" in config:
            test_datamodule = MEGDataModule(
                **config["test_datamodule_config"],
                seed=config["experiment"]["seed"],
            )
        else:
            test_datamodule = HarmonizationDataModule(
                **config["test_datamodule_config"],
                seed=config["experiment"]["seed"],
            )
        
    else:
        test_datamodule = datamodule

    trainer.test(datamodule=test_datamodule)
