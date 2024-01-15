import hashlib
import os
import warnings
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from mne_bids import (
    BIDSPath,
    read_raw_bids,
)

DATA_PATH = Path("/data/engs-pnpl/lina4368")


def _string_hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def load_dataset(bids_root, subject_id, task, session, preproc_config, cache_path=None):
    """Loads a dataset. First checks if a preprocessed version already exists in cache and loads it."""

    if not cache_path:
        # Generate a hash to describe this dataset and its processing configuration
        identifier = str(
            _string_hash(f"{bids_root}_{subject_id}_{session}_{task}_{preproc_config}")
        )
        cache_path = DATA_PATH / f"dataset_cache/{identifier}_raw.fif"
        (DATA_PATH / "dataset_cache").mkdir(parents=True, exist_ok=True)

        print(f"Computed dataset cache hash: {identifier}")

    # Load cached dataset if it exists, otherwise load BIDS dataset and preprocess
    if os.path.exists(cache_path):
        print("Cache found. Loading cached dataset.")
        raw = mne.io.read_raw_fif(cache_path, preload=False)  # Ensures lazy loading
        return raw, True, cache_path
    else:
        print("Cache not found. Loading dataset.")
        with warnings.catch_warnings():
            # Save the eyes from a huge trace of irrelevant warnings
            # http://tinyurl.com/yv3pnptu
            warnings.simplefilter("ignore")
            bids_path = BIDSPath(
                subject=subject_id,
                task=task,
                datatype="meg",
                suffix="meg",
                session=session,
                root=bids_root,
            )
            raw = read_raw_bids(bids_path)
        return raw, False, cache_path


def preprocess(raw, preproc_config, channels, cache_path):
    print(f"Preprocessing data with configuration {preproc_config}")

    if isinstance(channels, list):
        raw = raw.pick(channels)  # Pick only relevant MEG channels
    else:
        channels = [ch_name for ch_name in raw.ch_names]

    if preproc_config["filtering"]:
        raw.load_data()  # Warning: this loads all data into memory.
        raw.notch_filter(freqs=preproc_config["notch_freqs"], picks=channels)
        raw.filter(
            l_freq=preproc_config["bandpass_lo"],
            h_freq=preproc_config["bandpass_hi"],
            picks=channels,
        )

    if preproc_config["resample"]:
        raw = raw.resample(sfreq=preproc_config["resample"])

    # Cache processed data as a fif
    raw.save(cache_path, overwrite=True)

    return raw


def get_norm_stats(raw):
    # What kind of normalization should we apply for cross-subject + cross-dataset studies?
    # We should preserve: relative differences between channels
    # Simple solution: mean-center across channels + unit variance across channels
    # Account for outliers by min-max normalizing with 5th and 95th percentiles and mean-centering after this.

    print("Computing data normalization statistics")

    data = raw.get_data()  # Warning: this loads all data into memory.

    baseline_correction = data[:, :1000].mean(
        axis=1
    )  # Channel-wise mean from first 1000 samples

    data = (data.T - baseline_correction).T

    lq = np.percentile(data.flatten(), 0.25)
    uq = np.percentile(data.flatten(), 0.75)
    data = 2 * (((data) - lq) / (uq - lq)) - 1
    median = np.median(data.flatten())
    data -= median
    std = np.std(data.flatten())

    # Don't hold onto data to avoid keeping it in memory. Just pass the normalization statistics.
    del data

    return {
        "baseline_correction": baseline_correction,
        "median": median,
        "lower_q": lq,
        "upper_q": uq,
        "std": std,
    }


def normalize(data_slice, norm_stats):
    data_slice = (data_slice.T - norm_stats["baseline_correction"]).T
    data_slice = (
        2
        * (
            ((data_slice) - norm_stats["lower_q"])
            / (norm_stats["upper_q"] - norm_stats["lower_q"])
        )
        - 1
    )
    data_slice -= norm_stats["median"]
    data_slice = np.clip(
        data_slice, a_min=-20 * norm_stats["std"], a_max=20 * norm_stats["std"]
    )
    return data_slice


def get_slice(raw, idx, samples_per_slice):
    return raw[:, samples_per_slice * idx : samples_per_slice * (idx + 1)]


def get_slice_stats(raw, slice_len):
    duration = raw.times[-1] - raw.times[0]
    num_slices = int(duration / slice_len)
    samples_per_slice = int(int(raw.info["sfreq"]) * slice_len)
    return num_slices, samples_per_slice


class BatchScaler(nn.Module):
    """Applies scaling based on Defossez et al. 2023"""

    def __init__(self, correction_samples, n_sample_batches):
        super(BatchScaler, self).__init__()
        self.correction_samples = correction_samples
        self.n_sample_batches = n_sample_batches

    def fit(self, dataloader):
        sample_batches = []
        for i, batch in enumerate(dataloader):
            sample_batches.append(batch[0].cuda())  # Keep only data
            if i >= self.n_sample_batches:
                break

        data = torch.cat(sample_batches)

        self.baseline_correction = data.mean(dim=[0, 2])
        data = data - self.baseline_correction[None, :, None]

        self.std = torch.std(data)
        data = torch.clamp(data, min=20 * -self.std, max=20 * self.std)

        data = data.flatten()

        self.lower_q, self.median, self.upper_q = torch.quantile(
            data,
            q=torch.tensor([0.05, 0.5, 0.95], dtype=data.dtype, device=data.device),
        )

        data -= self.median
        self.lower_q = self.lower_q - self.median
        self.upper_q = self.upper_q - self.median

    def forward(self, batch):
        batch = batch - self.baseline_correction[None, :, None]
        batch = torch.clamp(batch, min=20 * -self.std, max=20 * self.std)
        batch -= self.median
        batch = 2 * ((batch - self.lower_q) / (self.upper_q - self.lower_q)) - 1
        return batch
