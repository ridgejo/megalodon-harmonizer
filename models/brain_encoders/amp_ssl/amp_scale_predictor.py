# Filter out a brain frequency band and predict which band was filtered

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TM


class AmpScalePredictor(nn.Module):
    """
    Given an encoded representation of the brain signal, predict which channel was masked with zeroes
    """

    def __init__(self, input_dim, hidden_dim, prop):
        super(AmpScalePredictor, self).__init__()

        self.prop = prop

        self.model = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=8,
            ),
        )

    def scale_amp(self, x):  # Assume x is [B, C, T]
        B, C, T = x.shape

        possible_scales = torch.linspace(start=-2, end=2, steps=8, device=x.device)
        scale_label = random.randrange(len(possible_scales))
        scale = possible_scales[scale_label]

        # Randomly select a proportion of the channels to apply the amplitude scaling to
        channels_to_scale = torch.randperm(C)[: int(C * self.prop)]

        x_scaled = x.clone()  # Avoids in-place gradient computation error
        x_scaled[:, channels_to_scale, :] *= scale

        return x_scaled, scale_label

    def forward(self, x, label):
        x = x.flatten(start_dim=1, end_dim=-1)  # [B, T, E] -> [B, T * E]
        z = self.model(x)

        label_tensor = torch.full((x.shape[0],), label, device=x.device)

        loss_ce = F.cross_entropy(z, label_tensor)

        probs = F.softmax(z, dim=1)
        preds = torch.argmax(probs, dim=1)

        accuracy = TM.classification.multiclass_accuracy(
            preds, label_tensor, num_classes=8
        )

        return {
            "amp_scale_ce_loss": loss_ce,
            "amp_scale_accuracy": accuracy,
        }
