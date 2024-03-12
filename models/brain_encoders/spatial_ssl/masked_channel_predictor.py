# Ideas for spatial sensor SSL (progressively increasing difficulty):
# - Mask (zero out) a channel and predict which channel
# - Circular shift a channel and predict which channel
# - Mask channel and regress the contents of the channel

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedChannelPredictor(nn.Module):
    """
    Given an encoded representation of the brain signal, predict which channel was masked with zeroes
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MaskedChannelPredictor, self).__init__()

        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(
            in_features=hidden_dim,
            out_features=1,
        )

    def mask_input(self, x):  # Assume x is [B, C, T]
        # Randomly mask channels in signal
        B, C, T = x.shape
        random_indices = torch.randint(0, C, (B,)).to(x.device)

        one_hot_mask = torch.zeros(B, C, device=x.device).scatter_(
            1, random_indices.unsqueeze(1), 1
        )
        one_hot_mask = 1 - one_hot_mask.unsqueeze(-1)
        result_tensor = x * one_hot_mask

        return result_tensor, random_indices

    def forward(self, masked_encoded, label):
        _, (h_n, _) = self.model(masked_encoded)
        z = self.classifier(h_n.reshape(masked_encoded.shape[0], -1)).squeeze(-1)

        # Division to account for approx number of sensors
        return F.mse_loss(z, label.float() / 250), torch.sqrt(
            F.mse_loss(z * 250, label.float())
        )
