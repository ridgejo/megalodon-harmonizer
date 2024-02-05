import hashlib
import os
import warnings
from pathlib import Path

# import mne
import numpy as np
import torch
import torch.nn as nn
import mne
# from mne_bids import (
#     BIDSPath,
#     read_raw_bids,
# )
from osl import preprocessing, utils
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

DATA_PATH = Path("/data/engs-pnpl/lina4368")

def get_scaler_hash(batch):
    return str([batch[-1][k][0] for k in batch[-1].keys()])

def _string_hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def load_dataset(bids_root, subject_id, task, session, preproc_config, cache_path=None):
    """Loads a dataset. First checks if a preprocessed version already exists in cache and loads it."""

    if not cache_path:
        # Work out cache path
        fname = f"sub-{subject_id}_ses-{session}_task-{task}_meg_preproc_raw.fif"
        cache_path = str(bids_root) + f"/preproc/sub-{subject_id}/sub-{subject_id}_ses-{session}_task-{task}_meg"
        cache_path = cache_path + "/" + fname
        print(f"Computed dataset cache path: {cache_path}")

    # Load cached dataset if it exists
    if os.path.exists(cache_path):
        print("Cache found. Loading cached dataset.")
        raw = mne.io.read_raw_fif(cache_path, preload=False)  # Ensures lazy loading
        return raw, True, cache_path
    else:
        raise FileNotFoundError(f"Could not find {cache_path}. Run preprocessing first.")

def get_valid_indices(raw, slice_len):
    """
    Takes a raw object and the slice length, generating a set of iterable indices that skip bad segments.
    """

    sfreq = float(raw.info["sfreq"])
    slice_samples = int(sfreq * slice_len)

    annotations = raw.annotations

    # Compute list of all indices
    valid_indices = set(range(0, len(raw), slice_samples))
    orig_size = len(valid_indices)

    # Compute indices of bad segments
    for annot in annotations:
        onset_samples = int(sfreq * annot["onset"])
        duration_samples = int(sfreq * annot["duration"])

        # Which index does the onset occur in?
        onset_idx = (onset_samples // slice_samples) * slice_samples

        # Which index does the bad segment end in?
        end_idx = ((onset_samples + duration_samples) // slice_samples) * slice_samples

        # Which indices should we ignore?
        ignore = set(range(onset_idx, end_idx + 1, slice_samples))

        valid_indices -= ignore

    valid_indices = sorted(list(valid_indices))[:-1] # Drop last to get even slice sizes.

    final_size = len(valid_indices)
    print(f"Initialised with {orig_size}, now {final_size}. Removed {orig_size - final_size} bad slices.")

    return valid_indices, slice_samples
        

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


def get_slice(raw, idx, samples_per_slice, valid_indices):
    sample_start = valid_indices[idx]
    return raw[:, sample_start : sample_start + samples_per_slice]


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
            self.scaler = QuantileTransformer(output_distribution="normal")
        elif "robust_scaler" in scaler_conf:
            self.scaler = RobustScaler(
                quantile_range=(
                    scaler_conf["robust_scaler"]["lo_q"],
                    scaler_conf["robust_scaler"]["hi_q"],
                ),
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
            data -= self.mean[None, :, None]  # baseline correction
            data = data.view(-1, 1)
            transformed = self.scaler.fit_transform(data)
            self.std = transformed.std()

    def forward(self, batch):
        if self.per_channel:
            batch = batch.permute(0, 2, 1)
            shape = batch.shape
            batch = batch.flatten(start_dim=0, end_dim=1)
            batch = np.clip(
                batch, a_min=self.low, a_max=self.high
            )  # Clip before transform
            batch = self.scaler.transform(batch)
            # batch = np.clip(batch, a_min=-self.std * 20, a_max=self.std * 20)
            return batch.reshape(*shape).transpose(0, 2, 1)
        else:
            batch -= self.mean[None, :, None]  # baseline correction
            shape = batch.shape
            batch = self.scaler.transform(batch.view(-1, 1))
            batch = np.clip(batch, a_min=-self.std * 20, a_max=self.std * 20)
            return batch.reshape(*shape)

if __name__ == "__main__":

    # Test preprocessed file.

    import mne
    raw = mne.io.read_raw_fif("/data/engs-pnpl/lina4368/armeni2022/preproc/sub-001/sub-001_ses-001_task-compr_meg/sub-001_ses-001_task-compr_meg_preproc_raw.fif", preload=False)
    raw = raw.pick_types(meg=True, ref_meg=False, exclude=["MRC23-4304", "MLO22-4304", "MRP5-4304", "MLC23-4304"])

    breakpoint()

    # batch_preprocess()