import os
import warnings
import hashlib
import numpy as np
from mne_bids import (
    BIDSPath,
    read_raw_bids,
)
from pathlib import Path
import pickle

DATA_PATH = Path('/data/engs-pnpl/lina4368')

def _string_hash(text):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)

def load_dataset(bids_root, subject_id, task, session, preproc_config):
    """"Loads a dataset. First checks if a preprocessed version already exists in cache and loads it."""

    # Generate a hash to describe this dataset and its processing configuration
    identifier = str(_string_hash(
        f"{bids_root}_{subject_id}_{session}_{task}_{preproc_config}")
    )
    cache_path = DATA_PATH / f"dataset_cache/{identifier}.pkl"
    (DATA_PATH / "dataset_cache").mkdir(parents=True, exist_ok=True)

    print(f"Computed dataset cache hash: {identifier}")

    # Load cached dataset if it exists, otherwise load BIDS dataset and preprocess
    if os.path.exists(cache_path):
        print("Cache found. Loading cached dataset.")
        with open(cache_path, 'rb') as f:
            raw = pickle.load(f)
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

    raw = raw.pick(channels) # Pick only relevant MEG channels

    if preproc_config["filtering"]:
        raw.load_data()
        raw.notch_filter(freqs=preproc_config["notch_freqs"], picks=channels)
        raw.filter(l_freq=preproc_config["bandpass_lo"], h_freq=preproc_config["bandpass_hi"], picks=channels)
    
    if preproc_config["resample"]:
        raw = raw.resample(sfreq=preproc_config["resample"])

    # Cache newly preprocessed data for faster loading next time.
    with open(cache_path, 'wb') as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return raw

def get_norm_stats(raw):
    # What kind of normalization should we apply for cross-subject + cross-dataset studies?
    # We should preserve: relative differences between channels
    # Simple solution: mean-center across channels + unit variance across channels
    # Account for outliers by min-max normalizing with 5th and 95th percentiles and mean-centering after this.

    print("Computing data normalization statistics")

    data = raw.get_data() # Warning: I think this loads all data into memory.
    
    p5 = np.percentile(data.flatten(), 0.05)
    p95 = np.percentile(data.flatten(), 0.95)
    mean = (2 * (((data) - p5) / (p95 - p5)) - 1).mean()

    # Don't hold onto data to avoid keeping it in memory. Just pass the normalization statistics.

    return mean, p5, p95

def normalize(data_slice, mean, p5, p95):
    data_slice = 2 * (((data_slice) - p5) / (p95 - p5)) - 1
    data_slice -= mean
    return data_slice

def get_slice(raw, idx, samples_per_slice):
    return raw[
        : , samples_per_slice * idx : samples_per_slice * (idx + 1)
    ]

def get_slice_stats(raw, slice_len):
    duration = raw.times[-1] - raw.times[0]
    num_slices = int(duration / slice_len)
    samples_per_slice = int(int(raw.info["sfreq"]) * slice_len)
    return num_slices, samples_per_slice