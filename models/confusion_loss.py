# Nicola Dinsdale 2020
# Define the loss function for the confusion part of the network
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
import numpy as np
########################################################################################################################

class ConfusionLoss(nn.Module):
    def __init__(self, task=0):
        super(ConfusionLoss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss