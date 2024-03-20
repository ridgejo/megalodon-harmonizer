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

    def __init__(self, input_dim, hidden_dim):
        super(MaskedChannelPredictor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=1,
            )
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

        x = masked_encoded.flatten(start_dim=1, end_dim=-1) # [B, T, E] -> [B, T * E]
        z = self.model(x).squeeze(-1)

        # Division to account for approx number of sensors
        mse = F.mse_loss(z, label.float() / 250)
        rmse = torch.sqrt(F.mse_loss(z * 250, label.float()))

        return {
            "masked_channel_mse_loss": mse,
            "masked_channel_rmse": rmse,
        }
