import torch.nn as nn


class Projector(nn.Module):
    """SSL projector (used in training, discarded during fine-tuning)"""

    def __init__(self, input_dim, hidden_dim):
        super(Projector, self).__init__()

        self.ssl_projector = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=input_dim,
            ),
        )

    def forward(self, z):
        return self.ssl_projector(z)
