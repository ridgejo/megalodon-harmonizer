import glob
import os

import torch
import torch.nn as nn

from dataloaders.data_utils import DATA_PATH
from dataloaders.data_module import DATASET_CLASSES


class SubjectEmbedding(nn.Module):
    """
    Differentiable subject-specific embeddings learned for each subject.
    """

    def __init__(self, dataset_keys: list, embedding_dim, freeze=False):
        """
        Args:
            subject_ids: A list of all subject IDs from all datasets
            in_channels:
            out_channels:
        """

        super(SubjectEmbedding, self).__init__()

        dataset_embeddings = {}
        for key in dataset_keys:
            dataset_embeddings[key] = nn.Embedding(
                num_embeddings=len(DATASET_CLASSES[key].subjects),
                embedding_dim=embedding_dim,
            )
        self.dataset_embeddings = nn.ModuleDict(dataset_embeddings)

    def forward(self, dataset_key, subject_ids):
        return self.dataset_embeddings[dataset_key](subject_ids)
