import torch.nn as nn


class DatasetLayer(nn.Module):
    """Differentiable dataset-specific transformation layer. Transforms all datasets into a shared space."""

    def __init__(self, dataset_sizes: dict, shared_dim=128, use_data_block=True):
        """
        Args:
            dataset_sizes: specifies input channels e.g., {"dataset id" : input channels}
            output_dim: shared output feature space
        """

        super(DatasetLayer, self).__init__()

        self.dataset_encoders = nn.ModuleDict(
            {
                dataset_id: nn.Conv1d(
                    in_channels=data_channels, out_channels=shared_dim, kernel_size=1
                ) if use_data_block else nn.Identity()
                for dataset_id, data_channels in dataset_sizes.items()
            }
        )

        self.dataset_decoders = nn.ModuleDict(
            {
                dataset_id: nn.Conv1d(
                    in_channels=shared_dim, out_channels=data_channels, kernel_size=1
                ) if use_data_block else nn.Identity()
                for dataset_id, data_channels in dataset_sizes.items()
            }
        )

    def forward(self, data, dataset_id):
        return self.dataset_encoders[dataset_id](data)

    def decode(self, data, dataset_id):
        return self.dataset_decoders[dataset_id](data)
