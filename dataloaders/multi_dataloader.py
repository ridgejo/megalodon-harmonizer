import lightning as L
import typing as tp
import sys

from torch.utils.data import ConcatDataset, DataLoader, random_split
from lightning.pytorch.utilities.combined_loader import CombinedLoader

class MultiDataLoader(L.LightningDataModule):
    """Loads data from multiple datasets. Supports loading mixed labelled and unlabelled data."""

    def __init__(
        self,
        dataset_preproc_configs : tp.Dict,
        dataloader_configs : tp.Dict,
    ):
        super().__init__()
        self.dataset_preproc_configs = dataset_preproc_configs
        self.dataloader_configs = dataloader_configs
        self.data = {}

        self.loaders = {
            "armeni2022": self._load_armeni_2022
        }
    
    def prepare_data(self):

        # Preprocess or load cached versions of all required datasets
        # NOTE: Called once within a single process on CPU

        for dataset, config in self.dataset_preproc_configs.items():

            data, seconds = self.loaders[dataset](config)
            self.data[dataset] = data

            print(
                f"Loaded approximately {seconds // 3600} hours of data from {dataset}."
            )

    def setup(self, stage: str):

        # Perform train/val/test splits on all datasets

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
            self.val[dataset] = Dataloader(val_split, batch_size=batch_size, shuffle=False, drop_last=False)
            self.test[dataset]= Dataloader(test_split, batch_size=batch_size, shuffle=False, drop_last=False)
            self.pred[dataset] = Dataloader(pred_split, batch_size=batch_size, shuffle=True, drop_last=True)

    def train_dataloader(self):

        # NOTE: 'max_size' will stop after the longest iterable is done, returning None for exhausted iterables.

        return CombinedLoader(self.train, 'max_size')

    def val_dataloader(self):
        return CombinedLoader(self.val, 'max_size')

    def test_dataloader(self):
        return CombinedLoader(self.test, 'max_size')
    
    def predict_dataloader(self):
        return CombinedLoader(self.predict, 'max_size')

    def _load_armeni_2022(self, config):

        bad_subjects = []

        # Loop over subjects
        for subj_no in range(1, n_subjects + 1):
            subject = "{:03d}".format(subj_no)  # 001, 002, and 003

            if subject in bad_subjects:
                continue

            # Loop over sessions
            for sess_no in range(1, n_sessions + 1):
                session = "{:03d}".format(sess_no)

                if session in bad_sessions[subject]:
                    continue

                if labels:
                    data = Armeni2022Labelled(
                        subject_id=subject,
                        session=session,
                        task="compr",
                        slice_len=slice_len,
                        preproc_config=preproc_config,
                        label_type=labels,
                    )
                else:
                    data = Armeni2022(
                        subject_id=subject,
                        session=session,
                        task="compr",  # There is only one relevant task (compr), the emptyroom task is irrelevant.
                        slice_len=slice_len,
                        preproc_config=preproc_config,
                    )

                seconds += len(data) * slice_len

                datasets.append(data)

        return datasets, seconds