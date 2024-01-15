import torch.nn as nn
import torch.nn.functional as F

from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock


def _make_test_mlp(
    vq_dim, codebook_size, shared_dim, hidden_dim, dataset_sizes, subject_ids
):
    return TestMLP()


class TestMLP(nn.Module):
    def __init__(self):
        super(TestMLP, self).__init__()

        shared_dim = 269
        compress_dim = 128

        self.enc = nn.Conv1d(
            in_channels=shared_dim, out_channels=compress_dim, kernel_size=1
        )

        self.dec = nn.Conv1d(
            in_channels=compress_dim, out_channels=shared_dim, kernel_size=1
        )

        self.dataset_layer = DatasetLayer(
            dataset_sizes={"Armeni2022": 269}, shared_dim=shared_dim
        )

        self.subject_block = SubjectBlock(
            subject_ids=["001"],
            in_channels=shared_dim,
            out_channels=shared_dim,
        )

        self.elu = nn.ELU()

    def forward(self, x, dataset_id, subject_id):
        # z = self.enc(x)
        # x_hat = self.dec(z)

        z = self.dataset_layer(x, dataset_id)
        # z = self.elu(z)
        z = self.subject_block(z, subject_id)
        # z = self.elu(z)
        z = self.enc(z)
        x_hat = self.dec(z)
        # x_hat = self.elu(x_hat)
        x_hat = self.subject_block.decode(x_hat, subject_id)
        # x_hat = self.elu(x_hat)
        x_hat = self.dataset_layer.decode(x_hat, dataset_id)

        loss = {
            "loss": F.mse_loss(x, x_hat),
        }

        return x_hat, loss
