import torch.nn as nn


class DatasetBlock(nn.Module):
    """Differentiable dataset-specific transformation layer. Transforms all datasets into a shared space."""

    def __init__(self, dataset_sizes: dict, shared_dim=128, use_data_block=True):
        """
        Args:
            dataset_sizes: specifies input channels e.g., {"dataset id" : input channels}
            output_dim: shared output feature space
        """

        super(DatasetBlock, self).__init__()

        self.use_data_block = use_data_block

        self.dataset_encoders = nn.ModuleDict(
            {
                dataset_id: nn.Conv1d(
                    in_channels=data_channels, out_channels=shared_dim, kernel_size=1
                )
                if use_data_block
                else nn.Identity()
                for dataset_id, data_channels in dataset_sizes.items()
            }
        )

        self.dataset_decoders = nn.ModuleDict(
            {
                dataset_id: nn.Conv1d(
                    in_channels=shared_dim, out_channels=data_channels, kernel_size=1
                )
                if use_data_block
                else nn.Identity()
                for dataset_id, data_channels in dataset_sizes.items()
            }
        )

    def forward(self, data, dataset_id, stage="encode"):
        # if stage == "encode":
        return self.dataset_encoders[dataset_id](data)
        # elif stage == "task":
        #     if self.use_data_block:
        #         weight = self.dataset_encoders[dataset_id].weight.clone()
        #         bias = self.dataset_encoders[dataset_id].bias.clone()
        #         return nn.functional.conv1d(data, weight, bias, stride=self.dataset_encoders[dataset_id].stride,
        #                                     padding=self.dataset_encoders[dataset_id].padding, 
        #                                     dilation=self.dataset_encoders[dataset_id].dilation,
        #                                     groups=self.dataset_encoders[dataset_id].groups)
        #     else:
        #         return self.dataset_encoders[dataset_id](data)

    def decode(self, data, dataset_id):
        return self.dataset_decoders[dataset_id](data)
