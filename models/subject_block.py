import glob
import os

import torch
import torch.nn as nn

from dataloaders.data_utils import DATA_PATH


class SubjectBlock(nn.Module):
    """
    Differentiable subject-specific embeddings learned for each subject.
    """

    def __init__(self, dataset_keys: list, block_dim):
        """
        Args:
            subject_ids: A list of all subject IDs from all datasets
            in_channels:
            out_channels:
        """

        super(SubjectBlock, self).__init__()

        # Automatically find dataset subjects
        subject_keys = {}
        for dataset_key in dataset_keys:
            subjects = sorted(
                [
                    os.path.basename(path).replace("sub-", "")
                    for path in glob.glob(str(DATA_PATH) + f"/{dataset_key}/sub-*")
                ]
            )
            subject_keys[dataset_key] = subjects

        self.subject_blocks = nn.ModuleDict(
            {
                dataset_key: nn.ModuleDict(
                    {
                        subject_key: nn.Conv1d(
                            in_channels=block_dim,
                            out_channels=block_dim,
                            kernel_size=1,
                        )
                        for subject_key in subject_keys[dataset_key]
                    }
                )
                for dataset_key in dataset_keys
            }
        )

    def forward(self, x, dataset_key, subject_key):
        # Apply block to input
        return self.subject_blocks[dataset_key][subject_key](x)
