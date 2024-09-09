import typing as tp

import lightning as L
import torch
from pnpl.dataloaders import MultiDataLoader
from pnpl.datasets import Armeni2022, Gwilliams2022, Schoffelen2019, Shafto2014
from torch.utils.data import DataLoader, random_split, RandomSampler
from .data_utils import ComboLoader, get_oversampler, Oversampler


DATASET_CLASSES = {
    "armeni2022": Armeni2022,
    "gwilliams2022": Gwilliams2022,
    "schoffelen2019": Schoffelen2019,
    "shafto2014": Shafto2014,
    "shaftoIntersection": Shafto2014
}


class MEGDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_preproc_configs: tp.Dict,
        dataloader_configs: tp.Dict,
        seed: int,
        debug: bool = False,
    ):
        super().__init__()
        self.batch_size = dataloader_configs["batch_size"]
        self.dataset_preproc_configs = dataset_preproc_configs
        self.dataloader_configs = dataloader_configs
        self.seed = seed
        self.debug = debug

    def setup(self, stage: str):
        train_loaders, val_loaders, test_loaders = [], [], []
        for dataset, config in self.dataset_preproc_configs.items():
            data = DATASET_CLASSES[dataset](**config)
            train, val, test = random_split(
                data,
                [
                    self.dataloader_configs["train_ratio"],
                    self.dataloader_configs["val_ratio"],
                    self.dataloader_configs["test_ratio"],
                ],
                generator=torch.Generator().manual_seed(self.seed),
            )

            if self.dataloader_configs.get("use_workers", False):
                train_loaders.append(
                    DataLoader(
                        train,
                        batch_size=self.batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=8,
                        persistent_workers=True,
                    )
                )
                val_loaders.append(
                    DataLoader(
                        val,
                        batch_size=self.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=8,
                        persistent_workers=True,
                    )
                )
            else:
                train_loaders.append(
                    DataLoader(
                        train,
                        batch_size=self.batch_size,
                        shuffle=True,
                        pin_memory=True,
                    )
                )
                val_loaders.append(
                    DataLoader(
                        val,
                        batch_size=self.batch_size,
                        shuffle=False,
                        pin_memory=True,
                    )
                )
            test_loaders.append(
                DataLoader(
                    test, batch_size=self.batch_size, shuffle=False, pin_memory=True
                )
            )

        self.train_loader = MultiDataLoader(train_loaders, shuffle=True)
        self.val_loader = MultiDataLoader(val_loaders, shuffle=False)
        self.test_loader = MultiDataLoader(test_loaders, shuffle=False)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Standard scale the batch before training
        mean = batch["data"].mean(dim=(0, -1))
        std = batch["data"].std(dim=(0, -1))
        batch["data"] = (
            (batch["data"] - mean[None, :, None]) / std[None, :, None]
        ).float()

        return batch

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader
    
class HarmonizationDataModule(L.LightningModule):
    def __init__(
        self,
        dataset_preproc_configs: tp.Dict,
        dataloader_configs: tp.Dict,
        seed: int,
        debug: bool = False,
    ):
        super().__init__()
        self.batch_size = dataloader_configs["batch_size"]
        self.dataset_preproc_configs = dataset_preproc_configs
        self.dataloader_configs = dataloader_configs
        self.seed = seed
        self.debug = debug

    def setup(self, stage: str):
        train_loaders, val_loaders, test_loaders = [], [], []

        splits = {}
        for dataset, config in self.dataset_preproc_configs.items():
            data = DATASET_CLASSES[dataset](**config)
            train, val, test = random_split(
                data,
                [
                    self.dataloader_configs["train_ratio"],
                    self.dataloader_configs["val_ratio"],
                    self.dataloader_configs["test_ratio"],
                ],
                generator=torch.Generator().manual_seed(self.seed),
            )
            splits[dataset] = (train, val, test)

        for dataset, split in splits.items():
            train, val, test = split
            if dataset == "shaftoIntersection":
                train_sampler = Oversampler(train, batch_size=self.batch_size)
                val_sampler = Oversampler(val, batch_size=self.batch_size)

                if self.dataloader_configs.get("use_workers", False):
                    train_loaders.append(
                        DataLoader(
                            train,
                            batch_size=self.batch_size,
                            # shuffle=True,
                            pin_memory=True,
                            num_workers=8,
                            persistent_workers=True,
                            sampler=train_sampler
                            # replacement=True
                        )
                    )
                    val_loaders.append(
                        DataLoader(
                            val,
                            batch_size=self.batch_size,
                            # shuffle=False,
                            pin_memory=True,
                            num_workers=8,
                            persistent_workers=True,
                            sampler=val_sampler
                            # replacement=True
                        )
                    )
                else:
                    train_loaders.append(
                        DataLoader(
                            train,
                            batch_size=self.batch_size,
                            # shuffle=True,
                            pin_memory=True,
                            sampler=train_sampler
                            # replacement=True
                        )
                    )
                    val_loaders.append(
                        DataLoader(
                            val,
                            batch_size=self.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            # sampler=val_sampler
                            # replacement=True
                        )
                    )
            else:
                if self.dataloader_configs.get("use_workers", False):
                    train_loaders.append(
                        DataLoader(
                            train,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=8,
                            persistent_workers=True,
                        )
                    )
                    val_loaders.append(
                        DataLoader(
                            val,
                            batch_size=self.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8,
                            persistent_workers=True,
                        )
                    )
                else:
                    train_loaders.append(
                        DataLoader(
                            train,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True,
                        )
                    )
                    val_loaders.append(
                        DataLoader(
                            val,
                            batch_size=self.batch_size,
                            shuffle=False,
                            pin_memory=True,
                        )
                    )
            test_loaders.append(
                DataLoader(
                    test, batch_size=self.batch_size, shuffle=False, pin_memory=True
                )
            )

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Standard scale the batch before training
        batches = list(batch)
        for batch_i in batches:
            mean = batch_i["data"].mean(dim=(0, -1))
            std = batch_i["data"].std(dim=(0, -1))
            batch_i["data"] = (
                (batch_i["data"] - mean[None, :, None]) / std[None, :, None]
            ).float()

        return tuple(batches)

    def train_dataloader(self):
        return ComboLoader(self.train_loaders)

    def val_dataloader(self):
        return ComboLoader(self.val_loaders)
    
    def test_dataloader(self):
        return ComboLoader(self.test_loaders)

    def predict_dataloader(self):
        return ComboLoader(self.test_loaders)


