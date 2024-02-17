"""Train classifier on phoneme labels."""

import argparse
import glob
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
import torch.nn.functional as F

import dataloaders.data_utils as data_utils
from dataloaders.pretraining import load_pretraining_data
from models.lstm_classifier import _make_lstm_classifier
from models.transformer_classifier import _make_transformer_classifier
from models.lstm_seq import _make_lstm_seq
from models.linear_seq import _make_linear_seq
from models.mlp_seq import _make_mlp_seq
from models.transformer_seq import _make_transformer_seq
from models.full_epoch_classifier import _make_full_epoch_classifier
from models.brain_encoders.ch_vqvae import _make_ch_vqvae
from models.brain_encoders.seanet_vqvae import _make_seanet_vqvae

parser = argparse.ArgumentParser(
    prog="MEGalodon-phonemes",
    description="Train a classifier on phoneme labels or VAD labels.",
)
parser.add_argument("--config", help="Path to config file (yaml)", required=True)
parser.add_argument("--name", help="Name for run", default=None)
parser.add_argument(
    "--debug", help="Faster debug mode", action="store_true", default=False
)
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())

# Setup
seed = config["experiment"]["seed"]
random.seed(seed)
np.random.seed(seed=seed)
torch.manual_seed(seed)

run = wandb.init(
    project=parser.prog,
    dir=data_utils.DATA_PATH / "wandb",
    config=config,
)

experiment_name = run.name
timestr = time.strftime("%Y%m%d-%H%M%S")
exp_folder_name = "exp_" + timestr + "_" + experiment_name
exp_folder = data_utils.DATA_PATH / f"experiments/{exp_folder_name}"
exp_folder.mkdir(parents=True, exist_ok=True)

if args.name:
    run.name = run.name + "+" + args.name

# load data
batch_size = config["data"]["batch_size"]
train_sampler, test_sampler, scalers = load_pretraining_data(
    preproc_config=config["data"]["preproc_config"],
    slice_len=config["data"]["slice_len"],
    train_ratio=config["data"]["train_ratio"],
    batch_size=batch_size,
    norm_config=config["data"]["norm"],
    debug=args.debug,
    labels=config["data"]["label_type"],
    exclude_subjects=config["data"]["exclude_subjects"],
)

print(f"Loaded train : test ({len(train_sampler)}, {len(test_sampler)})")

subjects = []
for k in config["data"]["preproc_config"].keys():
    subjects.extend(
        [
            os.path.basename(path).replace("sub-", "")
            for path in glob.glob(str(data_utils.DATA_PATH) + f"/{k}/sub-*")
        ]
    )

# load tokenizer if given
use_tokenizer = False
if "tokenizer" in config:

    use_tokenizer = True
    if "ch" in config["tokenizer"]:
        tok_type = "ch"
        print("Loading ch tokenizer")
        first_ds = next(iter(config["data"]["preproc_config"].keys()))
        tokenizer = _make_ch_vqvae(
            sampling_rate=config["data"]["preproc_config"][first_ds]["resample"],
            vq_dim=config["tokenizer"]["ch"]["vq_dim"],
            codebook_size=config["tokenizer"]["ch"]["codebook_size"],
            shared_dim=config["tokenizer"]["ch"]["shared_dim"],
            temporal_dim=config["tokenizer"]["ch"]["temporal_dim"],
            hidden_dim=config["tokenizer"]["ch"]["hidden_dim"],
            dataset_sizes=config["tokenizer"]["dataset_sizes"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["tokenizer"]["ch"],
            use_data_block="data_block" in config["tokenizer"]["ch"],
        )
    elif "seanet" in config["tokenizer"]:
        tok_type = "seanet"
        print("Loading SEANet tokenizer")
        tokenizer = _make_seanet_vqvae(
            vq_dim=config["tokenizer"]["seanet"]["vq_dim"],
            codebook_size=config["tokenizer"]["seanet"]["codebook_size"],
            shared_dim=config["tokenizer"]["seanet"]["shared_dim"],
            ratios=config["tokenizer"]["seanet"]["ratios"],
            conv_channels=config["tokenizer"]["seanet"]["conv_channels"],
            dataset_sizes=config["tokenizer"]["dataset_sizes"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["tokenizer"]["seanet"],
            use_data_block="data_block" in config["tokenizer"]["seanet"],
        )

    if "state_dict" in config["tokenizer"]:

        print("Loading state dictionary")

        tokenizer.load_state_dict(torch.load(config["tokenizer"]["state_dict"]))

        if "unfreeze" in config["tokenizer"]:
            print("Leaving tokenizer weights open to tuning")
        else:
            print("Freezing tokenizer weights")
            # By default, freeze tokenizer if weights are loaded
            for param in tokenizer.parameters():
                param.requires_grad = False
    
    tokenizer = tokenizer.cuda()


# load model
if config["data"]["label_type"] in ["voiced"]:

    if "lstm" in config["model"]:
        model = _make_lstm_classifier(
            dataset_sizes=config["data"]["dataset_sizes"],
            use_data_block="data_block" in config["model"]["lstm"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["lstm"],
            feature_dim=config["model"]["lstm"]["feature_dim"],
            hidden_dim=config["model"]["lstm"]["hidden_dim"],
            num_layers=config["model"]["lstm"]["num_layers"],
            output_classes=config["model"]["lstm"]["output_classes"],
        ).cuda()
    elif "transformer" in config["model"]:
        model = _make_transformer_classifier(
            dataset_sizes=config["data"]["dataset_sizes"],
            use_data_block="data_block" in config["model"]["transformer"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["transformer"],
            feature_dim=config["model"]["transformer"]["feature_dim"],
            hidden_dim=config["model"]["transformer"]["hidden_dim"],
            num_layers=config["model"]["transformer"]["num_layers"],
            output_classes=config["model"]["transformer"]["output_classes"],
        ).cuda()
    elif "full_epoch" in config["model"]:
        model = _make_full_epoch_classifier(
            dataset_sizes=config["data"]["dataset_sizes"],
            use_data_block="data_block" in config["model"]["full_epoch"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["full_epoch"],
            feature_dim=config["model"]["full_epoch"]["feature_dim"],
            time=config["model"]["full_epoch"]["time"],
            output_classes=config["model"]["full_epoch"]["output_classes"],
        ).cuda()

elif config["data"]["label_type"] in ["vad"]:

    # Project VQ dim to equivalent size if using tokenizer.
    if use_tokenizer:
        dataset_sizes = {k : config["tokenizer"][tok_type]["vq_dim"] for k in config["data"]["dataset_sizes"].keys()}
    else:
        dataset_sizes = config["data"]["dataset_sizes"]

    if "lstm" in config["model"]:
        model = _make_lstm_seq(
            dataset_sizes=dataset_sizes,
            use_data_block="data_block" in config["model"]["lstm"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["lstm"],
            feature_dim=config["model"]["lstm"]["feature_dim"],
            hidden_dim=config["model"]["lstm"]["hidden_dim"],
            num_layers=config["model"]["lstm"]["num_layers"],
            output_classes=config["model"]["lstm"]["output_classes"],
        ).cuda()
    elif "transformer" in config["model"]:
        model = _make_transformer_seq(
            dataset_sizes=dataset_sizes,
            use_data_block="data_block" in config["model"]["transformer"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["transformer"],
            feature_dim=config["model"]["transformer"]["feature_dim"],
            hidden_dim=config["model"]["transformer"]["hidden_dim"],
            num_layers=config["model"]["transformer"]["num_layers"],
            output_classes=config["model"]["transformer"]["output_classes"],
        ).cuda()
    elif "linear" in config["model"]:
        model = _make_linear_seq(
            dataset_sizes=dataset_sizes,
            use_data_block="data_block" in config["model"]["linear"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["linear"],
            feature_dim=config["model"]["linear"]["feature_dim"],
            output_classes=config["model"]["linear"]["output_classes"],
        ).cuda()
    elif "mlp" in config["model"]:
        model = _make_mlp_seq(
            dataset_sizes=dataset_sizes,
            use_data_block="data_block" in config["model"]["mlp"],
            subject_ids=subjects,
            use_sub_block="sub_block" in config["model"]["mlp"],
            feature_dim=config["model"]["mlp"]["feature_dim"],
            hidden_dim=config["model"]["mlp"]["hidden_dim"],
            output_classes=config["model"]["mlp"]["output_classes"],
        ).cuda()



# load optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["experiment"]["lr"])

num_epochs = config["experiment"]["epochs"]
iter_update_freq = config["stats"]["iter_update_freq"]
iteration = 0
start = time.perf_counter()
train_losses = Counter()
train_examples = {}
for epoch in range(num_epochs):
    for i, batch in enumerate(train_sampler):
        optimizer.zero_grad()

        x, label, times, dataset_id, subject_id = (
            batch[0],
            batch[1],
            batch[2],
            batch[-1]["dataset"][0],
            batch[-1]["subject"][0],
        )

        x = scalers[data_utils.get_scaler_hash(batch)](x)
        x = torch.from_numpy(x).cuda().float()

        label = label.cuda()

        if use_tokenizer:
            # [B, C, T] -> [B, E, Q]
            x = tokenizer.encode(x, dataset_id, subject_id)[0]

            # Resample label if sequence labelling
            if config["data"]["label_type"] in ["vad"]:
                label = F.interpolate(label.unsqueeze(1), size=x.shape[-1]).squeeze(1)

        y_hat, loss = model(x, label, dataset_id, subject_id)

        loss["loss"].backward()
        optimizer.step()

        train_losses.update(loss)

        # Log data in iterations due to huge amount of data in use.
        if iteration != 0 and iteration % iter_update_freq == 0:
            for k in train_losses:
                if k.startswith("D_"):
                    if "loss" in k or "score" in k or "acc" in k:
                        ds_id = k.split("_")[1]
                        train_losses[k] /= train_losses[f"D_{ds_id}"]
                elif k.startswith("S_"):
                    if "loss" in k or "score" in k or "acc" in k:
                        ks = k.split("_")
                        ds_id, sj_id = ks[1], ks[2]
                        train_losses[k] /= train_losses[f"S_{ds_id}_{sj_id}"]
                else:
                    train_losses[k] /= iter_update_freq

            with torch.no_grad():
                train_fig, test_fig = None, None
                test_losses = Counter()
                test_examples = {}
                for i, batch in enumerate(test_sampler):
                    x, label, times, dataset_id, subject_id = (
                        batch[0],
                        batch[1],
                        batch[2],
                        batch[-1]["dataset"][0],
                        batch[-1]["subject"][0],
                    )
                    x = scalers[data_utils.get_scaler_hash(batch)](x)
                    x = torch.from_numpy(x).cuda().float()
                    label = label.cuda()
                    if use_tokenizer:
                        x = tokenizer.encode(x, dataset_id, subject_id)[0]
                        if config["data"]["label_type"] in ["vad"]:
                            label = F.interpolate(label.unsqueeze(1), size=x.shape[-1]).squeeze(1)
                    y_hat, test_loss = model(x, label, dataset_id, subject_id)
                    test_losses.update(test_loss)

                for k in test_losses:
                    if k.startswith("D_"):
                        if "loss" in k or "score" in k or "acc" in k:
                            ds_id = k.split("_")[1]
                            test_losses[k] /= test_losses[f"D_{ds_id}"]
                    elif k.startswith("S_"):
                        if "loss" in k or "score" in k or "acc" in k:
                            ks = k.split("_")
                            ds_id, sj_id = ks[1], ks[2]
                            test_losses[k] /= test_losses[f"S_{ds_id}_{sj_id}"]
                    else:
                        test_losses[k] /= len(test_sampler)

                test_losses = {f"test_{k}": v for k, v in test_losses.items()}

            print()
            print(
                f"Epoch {round(iteration / len(train_sampler), 2)} Iter {iteration}/{num_epochs * len(train_sampler)}"
            )
            print(train_losses)
            print(test_losses)
            print()

            torch.save(model.state_dict(), exp_folder / f"{iteration}.pt")

            run.log(
                {
                    "iteration": iteration,
                    "epoch": iteration / len(train_sampler),
                    **train_losses,
                    **test_losses,
                    "epoch_time": time.perf_counter() - start,
                }
            )

            train_examples = {}
            train_losses = Counter()
            start = time.perf_counter()

        iteration += 1
