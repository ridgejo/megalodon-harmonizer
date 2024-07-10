# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn
import torch


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        # y, _ = self.lstm(x)

        # Clone the LSTM parameters to avoid in-place operations
        lstm_weights = []
        lstm_biases = []
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                lstm_weights.append(param.clone().to(x.device))
            elif 'bias' in name:
                lstm_biases.append(param.clone().to(x.device))

        # Separate weights and biases into groups for LSTM layers
        weight_ih = [lstm_weights[i] for i in range(0, len(lstm_weights), 2)]
        weight_hh = [lstm_weights[i] for i in range(1, len(lstm_weights), 2)]
        bias_ih = [lstm_biases[i] for i in range(0, len(lstm_biases), 2)]
        bias_hh = [lstm_biases[i] for i in range(1, len(lstm_biases), 2)]

        # Initialize hidden and cell states
        h_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.input_size, device=x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.input_size, device=x.device)
        
        # Manually call functional LSTM
        y, _ = nn.functional.lstm(x, (h_0, c_0), weight_ih, weight_hh, bias_ih, bias_hh)

        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
