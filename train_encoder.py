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

import dataloaders.data_utils as data_utils
from dataloaders.pretraining import load_pretraining_data
from models.brain_encoders.short_vqvae import _make_short_vqvae

parser = argparse.ArgumentParser(
    prog="MEGalodon-encoder",
    description="Train a tokenizer to encode brain signals",
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

# load data
batch_size = config["data"]["batch_size"]
train_sampler, test_sampler, scalers = load_pretraining_data(
    preproc_config=config["data"]["preproc_config"],
    slice_len=config["data"]["slice_len"],
    train_ratio=config["data"]["train_ratio"],
    batch_size=batch_size,
    baseline_correction_samples=1000,
    n_sample_batches=config["data"]["n_sample_batches"],
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

        x, times, dataset_id, subject_id = (
            batch[0].cuda(),
            batch[1],
            batch[2][0],
            batch[3][0],
        )

        x = scalers[dataset_id][subject_id](x).float()

        x_hat, loss = model(x, dataset_id, subject_id)

        loss["loss"].backward()
        optimizer.step()

        if dataset_id not in train_examples:
            train_examples[dataset_id] = (x, x_hat, times)

        train_losses.update(loss)

        # Log data in iterations due to huge amount of data in use.
        if iteration != 0 and iteration % iter_update_freq == 0:
            for k in train_losses:
                train_losses[k] /= iter_update_freq

            with torch.no_grad():
                train_fig, test_fig = None, None
                test_losses = Counter()
                test_examples = {}
                for i, batch in enumerate(test_sampler):
                    x, times, dataset_id, subject_id = (
                        batch[0].cuda(),
                        batch[1],
                        batch[2][0],
                        batch[3][0],
                    )
                    x = scalers[dataset_id][subject_id](x).float()
                    x_hat, test_loss = model(x, dataset_id, subject_id)
                    test_losses.update(test_loss)

                    if dataset_id not in test_examples:
                        test_examples[dataset_id] = (x, x_hat, times)

                for k in test_losses:
                    test_losses[k] /= len(test_sampler)

                test_losses = {f"test_{k}": v for k, v in test_losses.items()}

                if (not args.debug) or (args.debug and ((epoch + 1) % 500 == 0)):

                    ncols = len(test_examples.keys())
                    test_fig, test_axes = plt.subplots(
                        nrows=2, ncols=ncols, figsize=(4 * 5, 10)
                    )

                    for j, (dataset_id, (x, x_hat, times)) in enumerate(test_examples.items()):

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
                    ncols = len(train_examples.keys())
                    train_fig, train_axes = plt.subplots(
                        nrows=2, ncols=ncols, figsize=(4 * 5, 10)
                    )

                    for j, (dataset_id, (x, x_hat, times)) in enumerate(test_examples.items()):

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

            train_examples = {}
            train_losses = Counter()
            start = time.perf_counter()

        iteration += 1
