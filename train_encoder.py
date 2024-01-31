"""Train an encoder / tokenizer on unlabelled brain data."""

import argparse
import glob
import os
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm
from flatten_dict import flatten

import dataloaders.data_utils as data_utils
from dataloaders.pretraining import load_pretraining_data
from models.brain_encoders.short_vqvae import _make_short_vqvae
from models.brain_encoders.temp_spatial_vqvae import _make_temp_spatial_vqvae
from models.brain_encoders.attention_vqvae import _make_attention_vqvae
from models.brain_encoders.ch_vqvae import _make_ch_vqvae

parser = argparse.ArgumentParser(
    prog="MEGalodon-encoder",
    description="Train a tokenizer to encode brain signals",
)
parser.add_argument("--config", help="Path to config file (yaml)", required=True)
parser.add_argument("--name", help="Name for run", default=None)
parser.add_argument(
    "--debug", help="Faster debug mode", action="store_true", default=False
)
parser.add_argument(
    "--device", default="cuda"
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

if "short" in config["model"]:
    model = _make_short_vqvae(
        vq_dim=config["model"]["short"]["vq_dim"],
        codebook_size=config["model"]["short"]["codebook_size"],
        shared_dim=config["model"]["short"]["shared_dim"],
        hidden_dim=config["model"]["short"]["hidden_dim"],
        dataset_sizes=config["data"]["dataset_sizes"],
        subject_ids=subjects,
        use_sub_block="sub_block" in config["model"]["short"],
        use_data_block="data_block" in config["model"]["short"],
    ).to(args.device)
elif "temp_spatial" in config["model"]:
    first_ds = next(iter(config["data"]["preproc_config"].keys()))
    model = _make_temp_spatial_vqvae(
        sampling_rate=config["data"]["preproc_config"][first_ds]["resample"],
        vq_dim=config["model"]["temp_spatial"]["vq_dim"],
        codebook_size=config["model"]["temp_spatial"]["codebook_size"],
        shared_dim=config["model"]["temp_spatial"]["shared_dim"],
    ).to(args.device)
elif "attention" in config["model"]:
    first_ds = next(iter(config["data"]["preproc_config"].keys()))
    model = _make_attention_vqvae(
        sampling_rate=config["data"]["preproc_config"][first_ds]["resample"],
        vq_dim=config["model"]["attention"]["vq_dim"],
        codebook_size=config["model"]["attention"]["codebook_size"],
        shared_dim=config["model"]["attention"]["shared_dim"],
        temporal_dim=config["model"]["attention"]["temporal_dim"],
        transformer_dim=config["model"]["attention"]["transformer_dim"],
    ).to(args.device)
elif "ch" in config["model"]:
    first_ds = next(iter(config["data"]["preproc_config"].keys()))
    model = _make_ch_vqvae(
        sampling_rate=config["data"]["preproc_config"][first_ds]["resample"],
        vq_dim=config["model"]["ch"]["vq_dim"],
        codebook_size=config["model"]["ch"]["codebook_size"],
        shared_dim=config["model"]["ch"]["shared_dim"],
        temporal_dim=config["model"]["ch"]["temporal_dim"],
    ).to(args.device)


# load optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["experiment"]["lr"])

num_epochs = config["experiment"]["epochs"]
iter_update_freq = config["stats"]["iter_update_freq"]
iteration = 0
start = time.perf_counter()
train_losses = Counter()
train_examples = {}
pbar = tqdm(total=iter_update_freq)
for epoch in range(num_epochs):
    for i, batch in enumerate(train_sampler):
        optimizer.zero_grad()

        x, times, dataset_id, subject_id = (
            batch[0],
            batch[1],
            batch[-1]["dataset"][0],
            batch[-1]["subject"][0],
        )

        x = scalers[data_utils.get_scaler_hash(batch)](x)
        x = torch.from_numpy(x).to(args.device).float()

        x_hat, loss = model(x, dataset_id, subject_id)

        loss["loss"].backward()
        optimizer.step()

        if dataset_id not in train_examples:
            train_examples[dataset_id] = {subject_id: (x, x_hat, times)}
        elif (
            subject_id not in train_examples[dataset_id]
            and len(train_examples[dataset_id]) < 3
        ):
            train_examples[dataset_id][subject_id] = (x, x_hat, times)

        train_losses.update(loss)

        pbar.update(1)

        # Log data in iterations due to huge amount of data in use.
        if iteration != 0 and iteration % iter_update_freq == 0:
            for k in train_losses:
                if k.startswith("D_"):
                    if "loss" in k:
                        ds_id = k.split("_")[1]
                        train_losses[k] /= train_losses[f"D_{ds_id}"]
                elif k.startswith("S_"):
                    if "loss" in k:
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
                    x, times, dataset_id, subject_id = (
                        batch[0],
                        batch[1],
                        batch[-1]["dataset"][0],
                        batch[-1]["subject"][0],
                    )
                    x = scalers[data_utils.get_scaler_hash(batch)](x)
                    x = torch.from_numpy(x).to(args.device).float()
                    x_hat, test_loss = model(x, dataset_id, subject_id)
                    test_losses.update(test_loss)

                    if dataset_id not in test_examples:
                        test_examples[dataset_id] = {subject_id: (x, x_hat, times)}
                    elif (
                        subject_id not in test_examples[dataset_id]
                        and len(test_examples[dataset_id]) < 3
                    ):
                        test_examples[dataset_id][subject_id] = (x, x_hat, times)

                for k in test_losses:
                    if k.startswith("D_"):
                        if "loss" in k:
                            ds_id = k.split("_")[1]
                            test_losses[k] /= test_losses[f"D_{ds_id}"]
                    elif k.startswith("S_"):
                        if "loss" in k:
                            ks = k.split("_")
                            ds_id, sj_id = ks[1], ks[2]
                            test_losses[k] /= test_losses[f"S_{ds_id}_{sj_id}"]
                    else:
                        test_losses[k] /= len(test_sampler)

                test_losses = {f"test_{k}": v for k, v in test_losses.items()}

                if (not args.debug) or (
                    args.debug and (epoch % (iter_update_freq * 5) == 0)
                ):
                    test_examples = flatten(test_examples, reducer="underscore")
                    ncols = len(test_examples.keys())
                    test_fig, test_axes = plt.subplots(
                        nrows=2,
                        ncols=ncols,
                        figsize=(ncols * 5, 10),
                        squeeze=False,
                        num=1,
                        clear=True,
                    )

                    for j, (dataset_id, (x, x_hat, times)) in enumerate(
                        test_examples.items()
                    ):
                        x_sample = x[0].cpu()
                        x_hat_sample = x_hat[0].cpu()
                        t = times[0].cpu()

                        for ch_x, ch_x_hat in zip(x_sample, x_hat_sample):
                            test_axes[0, j].plot(t, ch_x)
                            test_axes[1, j].plot(t, ch_x_hat)
                            test_axes[1, j].set_xlabel("Time (s)")
                            test_axes[0, j].set_ylabel("Amplitude")
                            test_axes[0, j].set_ylim(-5, 5)
                            test_axes[1, j].set_ylim(-5, 5)

                        test_axes[0, j].set_title(dataset_id)

                    # Also draw train set
                    train_examples = flatten(train_examples, reducer="underscore")
                    ncols = len(train_examples.keys())
                    train_fig, train_axes = plt.subplots(
                        nrows=2,
                        ncols=ncols,
                        figsize=(ncols * 5, 10),
                        squeeze=False,
                        num=2,
                        clear=True,
                    )

                    for j, (dataset_id, (x, x_hat, times)) in enumerate(
                        test_examples.items()
                    ):
                        x_sample = x[0].cpu()
                        x_hat_sample = x_hat[0].cpu()
                        t = times[0].cpu()

                        for ch_x, ch_x_hat in zip(x_sample, x_hat_sample):
                            train_axes[0, j].plot(t, ch_x)
                            train_axes[1, j].plot(t, ch_x_hat)
                            train_axes[1, j].set_xlabel("Time (s)")
                            train_axes[0, j].set_ylabel("Amplitude")
                            train_axes[0, j].set_ylim(-5, 5)
                            train_axes[1, j].set_ylim(-5, 5)

                        train_axes[0, j].set_title(dataset_id)

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
                    "train_recon": wandb.Image(train_fig) if train_fig else None,
                    "test_recon": wandb.Image(test_fig) if test_fig else None,
                }
            )

            pbar.close()
            pbar = tqdm(total=iter_update_freq)
            train_examples = {}
            train_losses = Counter()
            start = time.perf_counter()

        iteration += 1
