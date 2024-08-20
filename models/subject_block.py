import glob
import os

import torch.nn as nn
import torch
from torch.nn import functional as F

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

    def forward(self, x, dataset_key, subject_key, stage="encode"):
        # Apply block to input
        if stage == "encode":
            return self.subject_blocks[dataset_key][subject_key](x)
        elif stage == "task":
            conv = self.subject_blocks[dataset_key][subject_key]
            bias = conv.bias.clone() if conv.bias is not None else None

            # Manually compute the weight from weight_g and weight_v
            weight_g = conv.weight_g.clone().view(-1, 1, 1).to(x.device)
            weight_v = conv.weight_v.clone().to(x.device)
            weight_norm = torch.norm(weight_v, dim=(1, 2), keepdim=True)
            weight = weight_v * (weight_g / weight_norm)

            # Perform the convolution manually
            x = F.conv1d(x, weight, bias, stride=conv.stride, padding=conv.padding,
                        dilation=conv.dilation, groups=conv.groups)
            return x
