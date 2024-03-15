import glob
import os

import torch
import torch.nn as nn

from dataloaders.data_utils import DATA_PATH


class SubjectEmbedding(nn.Module):
    """
    Differentiable subject-specific embeddings learned for each subject.
    """

    def __init__(self, dataset_keys: list, embedding_dim):
        """
        Args:
            subject_ids: A list of all subject IDs from all datasets
            in_channels:
            out_channels:
        """

        super(SubjectEmbedding, self).__init__()

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

        self.subject_embeddings = nn.ParameterDict(
            {
                dataset_key: nn.ParameterDict(
                    {
                        subject_key: nn.Parameter(
                            data=torch.randn(embedding_dim),
                            requires_grad=True,
                        )
                        for subject_key in subject_keys[dataset_key]
                    }
                )
                for dataset_key in dataset_keys
            }
        )

    def forward(self, dataset_key, subject_key):
        # Return embedding after expanding to batch size
        return self.subject_embeddings[dataset_key][subject_key]
