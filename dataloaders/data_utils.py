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

from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler


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

def get_slice(raw, idx, samples_per_slice):
    return raw[:, samples_per_slice * idx : samples_per_slice * (idx + 1)]


def get_slice_stats(raw, slice_len):
    duration = raw.times[-1] - raw.times[0]
    num_slices = int(duration / slice_len)
    samples_per_slice = int(int(raw.info["sfreq"]) * slice_len)
    return num_slices, samples_per_slice


class BatchScaler(nn.Module):
    """Applies scaling based on Defossez et al. 2023"""

    def __init__(self, n_sample_batches, per_channel, scaler_conf):
        super(BatchScaler, self).__init__()
        self.n_sample_batches = n_sample_batches
        if "quantile_transformer" in scaler_conf:
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif "robust_scaler" in scaler_conf:
            self.scaler = RobustScaler(
                quantile_range=(scaler_conf["robust_scaler"]["lo_q"], scaler_conf["robust_scaler"]["hi_q"]),
                unit_variance=True,
            )
        elif "standard_scaler" in scaler_conf:
            self.scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaler type")

        self.per_channel = per_channel

    def fit(self, dataloader):
        sample_batches = []
        for i, batch in enumerate(dataloader):
            sample_batches.append(batch[0])  # Keep only data
            if i >= self.n_sample_batches:
                break

        data = torch.cat(sample_batches)        

        if self.per_channel:
            data = data.permute(0, 2, 1).flatten(start_dim=0, end_dim=1)

            # Clip anything outside reasonable percentile before fitting (good for StandardScaler)
            self.low, self.high = np.quantile(data, [0.0001, 0.9999], axis=0)
            # std = data.std(axis=0)
            # mean = data.mean(axis=0)
            # self.low = mean - std * 10.0
            # self.high = mean + std * 10.0
            data = np.clip(data, a_min=self.low, a_max=self.high)

            transformed = self.scaler.fit_transform(data)
            self.std = transformed.std(axis=0)
        else:
            self.mean = data.mean(axis=[0, 2])
            data -= self.mean[None, :, None] # baseline correction
            data = data.view(-1, 1)
            transformed = self.scaler.fit_transform(data)
            self.std = transformed.std()

    def forward(self, batch):

        if self.per_channel:
            batch = batch.permute(0, 2, 1)
            shape = batch.shape
            batch = batch.flatten(start_dim=0, end_dim=1)
            batch = np.clip(batch, a_min=self.low, a_max=self.high) # Clip before transform
            batch = self.scaler.transform(batch)
            # batch = np.clip(batch, a_min=-self.std * 20, a_max=self.std * 20)
            return batch.reshape(*shape).transpose(0, 2, 1)
        else:
            batch -= self.mean[None, :, None] # baseline correction
            shape = batch.shape
            batch = self.scaler.transform(batch.view(-1, 1))
            batch = np.clip(batch, a_min=-self.std * 20, a_max=self.std * 20)
            return batch.reshape(*shape)
