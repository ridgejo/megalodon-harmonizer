# Randomly Phase shift a randomly selected subset of channels and predict the phase shift

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TM


class PhaseDiffPredictor(nn.Module):
    """
    Given an encoded representation of the brain signal, predict which channel was masked with zeroes
    """

    def __init__(self, input_dim, hidden_dim, prop):
        super(PhaseDiffPredictor, self).__init__()

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

    def apply_random_phase_shift(self, x):  # Assume x is [B, C, T]
        B, C, T = x.shape

        # Randomly determine the phase shift
        possible_shifts = torch.linspace(
            start=0, end=math.pi - 0.25 * math.pi, steps=8, device=x.device
        )
        phase_shift_label = random.randrange(len(possible_shifts))
        phase_shift = possible_shifts[phase_shift_label]

        # FFT to convert the signal to frequency domain, using full FFT
        freq_x = torch.fft.fft(x, dim=2)

        # Calculate the phase shift factor in the complex plane
        phase_shift_factor = torch.exp(phase_shift * 1j)

        # Randomly select a proportion of the channels to apply the phase shift
        channels_to_shift = torch.randperm(C)[: int(C * self.prop)]

        # Apply the phase shift to the randomly selected channels
        freq_x[:, channels_to_shift, :] *= phase_shift_factor

        # Inverse FFT to convert back to time domain, using full IFFT
        time_x = torch.fft.ifft(
            freq_x, dim=2
        ).real  # Taking the real part since the original signal is real

        return time_x, phase_shift_label

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
            "phase_diff_ce_loss": loss_ce,
            "phase_diff_accuracy": accuracy,
        }
