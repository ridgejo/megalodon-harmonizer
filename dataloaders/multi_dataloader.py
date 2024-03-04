import lightning as L
import typing as tp
import sys

from torch.utils.data import ConcatDataset, DataLoader, random_split
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from dataloaders.data_utils import BatchScaler
from dataloaders.armeni2022 import Armeni2022

class MultiDataLoader(L.LightningDataModule):
    """Loads data from multiple datasets. Supports loading mixed labelled and unlabelled data."""

    def __init__(
        self,
        dataset_preproc_configs : tp.Dict,
        dataloader_configs : tp.Dict,
        debug : bool = False,
    ):
        super().__init__()
        self.dataset_preproc_configs = dataset_preproc_configs
        self.dataloader_configs = dataloader_configs
        self.debug = debug
        self.data = {}

        self.loaders = {
            "armeni2022": self._load_armeni_2022
        }
    
    def prepare_data(self):

        # Preprocess or load cached versions of all required datasets
        # NOTE: Called once within a single process on CPU

        for dataset, config in self.dataset_preproc_configs.items():

            print(f"Loading data from {dataset}...")

            data, seconds = self.loaders[dataset](config)
            self.data.update(data)

            print(
                f"Loaded approximately {seconds // 3600} hours of data from {dataset}"
            )

    def setup(self, stage: str):

        print("Performing train/val/test/pred splitting...")

        train_ratio, val_ratio, test_ratio, pred_ratio = (
            self.dataloader_configs["train_ratio"],
            self.dataloader_configs["val_ratio"],
            self.dataloader_configs["test_ratio"],
            self.dataloader_configs["pred_ratio"]
        )

        batch_size = self.dataloader_configs["batch_size"]

        self.train, self.val, self.test, self.pred = {}, {}, {}, {}
        for dataset, data in self.data.items():

            train_size = int(train_ratio * len(data))
            val_size = int(val_ratio * len(data))
            test_size = int(test_ratio * len(data))
            pred_size = len(data) - train_size - val_size - test_size

            if min([train_size, val_size, test_size, pred_size]) < batch_size:
                print(f"Warning: One of train/val/test/pred smaller than batch size {batch_size} for dataset {dataset}. Zero batches will be available.")

            train_split, val_split, test_split, pred_split = random_split(
                data, [train_size, val_size, test_size, pred_size]
            )

            self.train[dataset] = DataLoader(train_split, batch_size=batch_size, shuffle=True, drop_last=True)
            self.val[dataset] = DataLoader(val_split, batch_size=batch_size, shuffle=False, drop_last=False)
            self.test[dataset]= DataLoader(test_split, batch_size=batch_size, shuffle=False, drop_last=False)
            self.pred[dataset] = DataLoader(pred_split, batch_size=batch_size, shuffle=True, drop_last=True)
        
        print("Fitting scalers to datasets...")

        self.scalers = {}
        norm_conf = self.dataloader_configs["normalisation"]
        for dataset, train_dl in self.train.items():
            scaler = BatchScaler(
                n_sample_batches=norm_conf["n_sample_batches"],
                per_channel=norm_conf["per_channel"],
                scaler_conf=norm_conf["scaler_conf"],
            )
            scaler.fit(train_dl)
            self.scalers[dataset] = scaler

    def on_before_batch_transfer(self, batch, dataloader_idx):

        # Apply batch scaling transformation before transferring to device.
        for dataset, batch_tensor in batch.items():
            batch[dataset][0] = self.scalers[dataset](batch_tensor[0])

        return batch

    def train_dataloader(self):

        # NOTE: 'max_size' will stop after the longest iterable is done, returning None for exhausted iterables.

        return CombinedLoader(self.train, 'max_size')

    def val_dataloader(self):
        return CombinedLoader(self.val, 'max_size')

    def test_dataloader(self):
        return CombinedLoader(self.test, 'max_size')
    
    def predict_dataloader(self):
        return CombinedLoader(self.predict, 'max_size')

    def _load_armeni_2022(self, config, n_subjects=27, n_sessions=10):

        # Dataset key formatted as dat={}_sub={}_ses={}. Necessary for scalers.

        if self.debug:
            n_subjects = 1
            n_sessions = 1

        bad_subjects = config["bad_subjects"]
        bad_sessions = config["bad_sessions"]
        slice_len = config["slice_len"]
        label_type = config["label_type"]
        preproc_config = config["preproc_config"]

        datasets = {}

        # Loop over subjects
        seconds = 0
        for subj_no in range(1, n_subjects + 1):

            subject = "{:03d}".format(subj_no)  # 001, 002, and 003

            if subject in bad_subjects:
                continue

            # Loop over sessions
            for sess_no in range(1, n_sessions + 1):
                session = "{:03d}".format(sess_no)

                if session in bad_sessions[subject]:
                    continue

                data = Armeni2022(
                    subject_id=subject,
                    session=session,
                    task="compr",
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                    label_type=label_type,
                )

                seconds += len(data) * slice_len

                datasets[f"dat=armeni2022_sub={subject}_ses={session}"] = data

        return datasets, seconds

if __name__ == "__main__":

    datamodule = MultiDataLoader(
        dataset_preproc_configs={
            "armeni2022": {
                "bad_subjects": [],
                "bad_sessions": {"001": [], "002": [], "003": []},
                "slice_len": 3.0,
                "label_type": "vad",
                "preproc_config": {
                    "filtering": True,
                    "resample": 250,
                    "notch_freqs": [50, 100],
                    "bandpass_lo": 0.5,
                    "bandpass_hi": 125,
                },
            },
        },
        dataloader_configs={
            "train_ratio": 0.9,
            "val_ratio": 0.04,
            "test_ratio": 0.04,
            "pred_ratio": 0.02,
            "batch_size": 32,
            "normalisation": {
                "n_sample_batches": 8,
                "per_channel": True,
                "scaler_conf": {
                    "standard_scaler": None
                },
            },
        },
        debug=True,
    )

    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    sample = next(iter(datamodule.train_dataloader()))[0]
    print(sample)

    sample_scaled = datamodule.on_before_batch_transfer(sample, 0)
    print(sample_scaled)