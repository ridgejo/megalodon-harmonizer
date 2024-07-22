# Nicola Dinsdale 2020
# Define the loss function for the confusion part of the network
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
import numpy as np
########################################################################################################################

class ConfusionLoss(nn.Module):
    def __init__(self, task=0, epsilon=1e-6):
        super(ConfusionLoss, self).__init__()
        self.task = task
        self.epsilon = epsilon

    def forward(self, x, target):
        # Check for NaNs or Infs in input
        if torch.isnan(x).any():
            raise ValueError("NaN detected in input to ConfusionLoss")
        if torch.isinf(x).any():
            raise ValueError("Inf detected in input to ConfusionLoss")

        # Add epsilon to x to avoid log(0)
        x = x + self.epsilon

        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)

        # Check for NaNs or Infs in output
        if torch.isnan(loss).any():
            raise ValueError("NaN detected in input to ConfusionLoss")
        if torch.isinf(loss).any():
            raise ValueError("Inf detected in input to ConfusionLoss")

        return loss