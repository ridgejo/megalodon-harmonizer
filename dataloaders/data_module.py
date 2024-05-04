import typing as tp

import lightning as L
import torch
from pnpl.dataloaders import MultiDataLoader
from pnpl.datasets import Armeni2022, Gwilliams2022, Schoffelen2019
from torch.utils.data import DataLoader, random_split

DATASET_CLASSES = {
    "armeni2022": Armeni2022,
    "gwilliams2022": Gwilliams2022,
    "schoffelen2019": Schoffelen2019,
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

            # No pinning, no workers, not persistent: 0.3 it/s (4.0 again after cache hit)
            # Pinning, 16 workers, persistent: X
            # Pinning, 8 workers, persistent: 3.45 it/s
            # Pinning, 4 workers, persistent: 3.98 it/s
            # Pinning, 2 workers, persistent: 4.02 it/s
            # Pinning, 1 workers, persistent: 3.98 it/s
            # Pinning, 0 workers: 4.10 it/s
            # Conclusion: pinning is all you need? NO. Caches are all you need ;)
            # Also: for much larger datasets, workers actually become useful

            train_loaders.append(
                DataLoader(
                    train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    # num_workers=8,
                    # persistent_workers=True,
                )
            )
            val_loaders.append(
                DataLoader(
                    val,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    # num_workers=8,
                    # persistent_workers=True,
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
