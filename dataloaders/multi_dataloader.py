import glob
import os
import typing as tp

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from dataloaders.armeni2022 import Armeni2022
from dataloaders.batch_invariant_sampler import BatchInvariantSampler
from dataloaders.data_utils import DATA_PATH, BatchScaler
from dataloaders.gwilliams2022 import Gwilliams2022
from dataloaders.schoffelen2019 import Schoffelen2019


def get_key_from_batch_identifier(batch_identifier: dict) -> str:
    identifier = {k: batch_identifier[k][0] for k in batch_identifier.keys()}
    return get_key_from_identifier(identifier)


def get_key_from_identifier(identifier: dict) -> str:
    key = f"dat={identifier['dataset']}"
    if "subject" in identifier:
        key += f"_sub={identifier['subject']}"
    return key


class MultiDataLoader(L.LightningDataModule):
    """Loads data from multiple datasets. Supports loading mixed labelled and unlabelled data."""

    def __init__(
        self,
        dataset_preproc_configs: tp.Dict,
        dataloader_configs: tp.Dict,
        debug: bool = False,
    ):
        super().__init__()
        self.dataset_preproc_configs = dataset_preproc_configs
        self.dataloader_configs = dataloader_configs
        self.debug = debug
        self.data = {}

        self.loaders = {
            "armeni2022": self._load_armeni_2022,
            "gwilliams2022": self._load_gwilliams_2022,
            "schoffelen2019": self._load_schoffelen_2019,
        }

    def prepare_data(self):
        # Preprocess or load cached versions of all required datasets
        # NOTE: Called once within a single process on CPU

        for dataset, config in self.dataset_preproc_configs.items():
            print(f"Loading data from {dataset}...")

            data, seconds = self.loaders[dataset](config)
            self.data[dataset] = data

            print(
                f"Loaded approximately {seconds // 3600} hours of data from {dataset}"
            )

    def setup(self, stage: str):
        print("Performing train/val/test/pred splitting...")

        train_ratio, val_ratio, test_ratio = (
            self.dataloader_configs["train_ratio"],
            self.dataloader_configs["val_ratio"],
            self.dataloader_configs["test_ratio"],
        )

        batch_size = self.dataloader_configs["batch_size"]

        self.train, self.val, self.test, self.pred = {}, {}, {}, {}
        self.scalers = {}
        for dataset, datasets in self.data.items():
            # Data will be a list of datasets by subject and (possibly) session for each underlying dataset

            for data in datasets:

                if self.debug:
                    # Fit only one batch in debug mode
                    train_size = batch_size
                    val_size = batch_size
                    test_size = 1
                    pred_size = len(data) - train_size - val_size - test_size
                else:
                    train_size = int(train_ratio * len(data))
                    val_size = int(val_ratio * len(data))
                    test_size = int(test_ratio * len(data))
                    pred_size = len(data) - train_size - val_size - test_size

                if min([train_size, val_size]) < batch_size:
                    print(
                        f"Warning: One of train/val smaller than batch size {batch_size} for dataset {dataset}. Zero batches will be available."
                    )

                if min([test_size, pred_size]) < batch_size:
                    print(
                        f"Warning: One of pred/test smaller than batch size {batch_size} for dataset {dataset}. Zero batches will be available."
                    )

                train_split, val_split, test_split, pred_split = random_split(
                    data, [train_size, val_size, test_size, pred_size]
                )

                identifier = get_key_from_identifier(train_split[0]["identifier"])

                self.train[identifier] = DataLoader(
                    train_split, batch_size=batch_size, shuffle=True, drop_last=True
                )
                self.val[identifier] = DataLoader(
                    val_split, batch_size=batch_size, shuffle=False, drop_last=False
                )
                self.test[identifier] = DataLoader(
                    test_split, batch_size=batch_size, shuffle=False, drop_last=False
                )
                self.pred[identifier] = DataLoader(
                    pred_split, batch_size=batch_size, shuffle=False, drop_last=False
                )

        print("Fitting scalers to datasets...")

        self.scalers = {}
        norm_conf = self.dataloader_configs["normalisation"]
        for identifier, train_dl in self.train.items():
            scaler = BatchScaler(
                n_sample_batches=norm_conf["n_sample_batches"],
                per_channel=norm_conf["per_channel"],
                scaler_conf=norm_conf["scaler_conf"],
            )
            scaler.fit(train_dl)
            self.scalers[identifier] = scaler

        # Construct batch-invariant samplers
        self.train = BatchInvariantSampler(
            dataloaders=list(self.train.values()),
            shuffle=True,
        )
        self.val = BatchInvariantSampler(
            dataloaders=list(self.val.values()),
            shuffle=False,
        )
        self.test = BatchInvariantSampler(
            dataloaders=list(self.test.values()),
            shuffle=False,
        )
        self.pred = BatchInvariantSampler(
            dataloaders=list(self.pred.values()),
            shuffle=False,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # Get identifier from first sample
        key = get_key_from_batch_identifier(batch["identifier"])

        # Apply batch scaling transformation before transferring to device.
        batch["data"] = torch.from_numpy(self.scalers[key](batch["data"])).float()

        return batch

    def train_dataloader(self):
        return self.train  # Automatically balances datasets

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test

    def predict_dataloader(self):
        return self.pred

    def _load_armeni_2022(self, config, n_subjects=3, n_sessions=10):
        # Return list of datasets. Each dataset should correspond to a single subject and session.

        if self.debug:
            n_subjects = 1
            n_sessions = 1

        bad_subjects = config["bad_subjects"]
        bad_sessions = config["bad_sessions"]
        slice_len = config["slice_len"]
        label_type = config["label_type"]

        datasets = []

        # Loop over subjects
        seconds = 0
        for subj_no in range(1, n_subjects + 1):
            subject = "{:03d}".format(subj_no)  # 001, 002, and 003

            if subject in bad_subjects:
                continue

            # Loop over sessions
            sess_datasets = []
            for sess_no in range(1, n_sessions + 1):
                session = "{:03d}".format(sess_no)

                if session in bad_sessions[subject]:
                    continue

                data = Armeni2022(
                    subject_id=subject,
                    session=session,
                    task="compr",
                    slice_len=slice_len,
                    label_type=label_type,
                )

                seconds += len(data) * slice_len

                sess_datasets.append(data)

            datasets.append(ConcatDataset(sess_datasets))

        return datasets, seconds

    def _load_gwilliams_2022(self, config, n_subjects=27, n_sessions=2, n_tasks=3):

        if self.debug:
            n_subjects = 1

        bad_subjects = config["bad_subjects"]
        slice_len = config["slice_len"]
        label_type = config["label_type"]

        seconds = 0
        datasets = []
        for subj_no in range(1, n_subjects + 1):
            subject = "{:02d}".format(subj_no)  # 01, 02, etc.

            if subject in bad_subjects:
                continue

            # Loop over sessions
            sess_datasets = [] # Combine sessions in normalization as not much data available per subject
            for sess_no in range(0, n_sessions):
                session = str(sess_no)

                for task in range(0, n_tasks + 1):
                    task = str(task)

                    try:
                        data = Gwilliams2022(
                            subject_id=subject,
                            session=session,
                            task=task,
                            slice_len=slice_len,
                            label_type=label_type,
                        )
                    except Exception as e:
                        print("Skipping: ", e)
                        continue  # Subject may not have completed task or session

                    seconds += len(data) * slice_len

                    sess_datasets.append(data)

                if len(sess_datasets) > 0:
                    datasets.append(ConcatDataset(sess_datasets))

        return datasets, seconds

    def _load_schoffelen_2019(self, config, tasks=["auditory", "rest", "visual"]):
        subjects = sorted(
            [
                os.path.basename(path).replace("sub-", "")
                for path in glob.glob(str(DATA_PATH) + "/schoffelen2019/sub-*")
            ]
        )

        if self.debug:
            subjects = [subjects[0]]
            tasks = ["rest"]

        bad_subjects = config["bad_subjects"]
        slice_len = config["slice_len"]

        seconds = 0
        datasets = []
        for subject in subjects:
            if subject in bad_subjects:
                continue  # ignore incomplete subject data

            task_datasets = []
            for task in tasks:
                if subject.startswith("V") and task == "auditory":
                    continue
                elif subject.startswith("A") and task == "visual":
                    continue

                try:
                    data = Schoffelen2019(
                        subject_id=subject,
                        task=task,
                        slice_len=slice_len,
                    )
                except Exception as e:
                    print("Skipping: ", e)
                    continue  # Not all subjects have recordings for both tasks

                seconds += len(data) * slice_len

                task_datasets.append(data)

            if len(task_datasets) > 0:
                datasets.append(ConcatDataset(task_datasets))

        return datasets, seconds


if __name__ == "__main__":
    datamodule = MultiDataLoader(
        dataset_preproc_configs={
            "armeni2022": {
                "bad_subjects": [],
                "bad_sessions": {"001": [], "002": [], "003": []},
                "slice_len": 0.1,
                "label_type": "vad",
            },
            "schoffelen2019": {
                "bad_subjects": [],
                "slice_len": 0.1,
                "label_type": None,
            },
            "gwilliams2022": {
                "bad_subjects": [],
                "slice_len": 0.1,
                "label_type": None,
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
                "scaler_conf": {"standard_scaler": None},
            },
        },
        debug=True,
    )

    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    sample = next(iter(datamodule.train_dataloader()))
    print(sample)

    sample_scaled = datamodule.on_before_batch_transfer(sample, 0)
    print(sample_scaled)

    breakpoint()
