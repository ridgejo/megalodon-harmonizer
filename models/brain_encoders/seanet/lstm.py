# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn
import torch
import warnings

# Suppress the specific RNN warning
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk of memory.*")

class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x, stage="encode"):
        x = x.permute(2, 0, 1)
        # if stage == "encode":
        y, _ = self.lstm(x)
        # elif stage == "task":
        #     # Clone the LSTM parameters to avoid in-place operations
        #     lstm_params = []
        #     for name, param in self.lstm.named_parameters():
        #         lstm_params.append(param.clone())

        #     # Initialize hidden and cell states
        #     h_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.hidden_size, device=x.device)
        #     c_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.hidden_size, device=x.device)
            
        #     # Manually call functional LSTM using torch._VF.lstm
        #     y, _, _ = torch._VF.lstm(
        #         x, 
        #         (h_0, c_0), 
        #         lstm_params, 
        #         self.lstm.bias, 
        #         self.lstm.num_layers, 
        #         self.lstm.dropout, 
        #         self.lstm.training, 
        #         self.lstm.bidirectional, 
        #         False  # batch_first
        #     )
        
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
