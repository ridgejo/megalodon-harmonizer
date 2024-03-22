# Filter out a brain frequency band and predict which band was filtered

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TA
import torchmetrics.functional as TM
import random


class BandPredictor(nn.Module):
    """
    Given an encoded representation of the brain signal, predict which channel was masked with zeroes
    """

    def __init__(self, input_dim, hidden_dim):
        super(BandPredictor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=6,
            ),
        )

    def mask_input(self, x, sample_rate):  # Assume x is [B, C, T]

        # todo: filter different bands for each sample in batch? not possible without an intensive for-loop?

        B, C, T = x.shape

        # Pick band to mask at random
        band = random.randrange(6)

        bands = [
            (0.1, 3.0), # Delta
            (3.0, 8.0), # Theta
            (8.0, 12.0), # Alpha
            (12.0, 30.0) # Beta
            (30.0, 125.0) # Gamma
        ]

        low_cutoff = bands[band][0]
        high_cutoff = bands[band][1]

        # Calculate the center frequency and Q factor
        central_freq = (low_cutoff + high_cutoff) / 2.0  # Midpoint
        bandwidth = high_cutoff - low_cutoff  # Range to reject
        Q = central_freq / bandwidth

        # Apply the band-reject filter
        filtered = TA.bandreject_biquad(x, sample_rate, central_freq, Q)

        return filtered, band

    def forward(self, x, label):
        x = x.flatten(start_dim=1, end_dim=-1)  # [B, T, E] -> [B, T * E]
        z = self.model(x)

        label_tensor = torch.full((x.shape[0],), label)

        loss_ce = F.cross_entropy(z, label_tensor)

        probs = F.softmax(z, dim=1)
        preds = torch.argmax(probs, dim=1)

        accuracy = TM.classification.multiclass_accuracy(
            preds, label_tensor, num_classes=6
        )


        return {
            "band_ce_loss": loss_ce,
            "band_accuracy": accuracy,
        }
