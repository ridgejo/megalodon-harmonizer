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
                lstm_weights.append(param.clone())
            elif 'bias' in name:
                lstm_biases.append(param.clone())

        # Separate weights and biases into groups for LSTM layers
        weight_ih = [lstm_weights[i] for i in range(0, len(lstm_weights), 2)]
        weight_hh = [lstm_weights[i] for i in range(1, len(lstm_weights), 2)]
        bias_ih = [lstm_biases[i] for i in range(0, len(lstm_biases), 2)]
        bias_hh = [lstm_biases[i] for i in range(1, len(lstm_biases), 2)]

        # Initialize hidden and cell states
        h_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.input_size, device=x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.input_size, device=x.device)
        
        # Manually call functional LSTM
        y, _ = torch._VF.lstm(x, (h_0, c_0), weight_ih, weight_hh, bias_ih, bias_hh)

        # # Initialize hidden and cell states
        # h_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.input_size, device=x.device)
        # c_0 = torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.input_size, device=x.device)
        
        # # Extract LSTM parameters and clone them
        # weight_ih_l = [param.clone() for name, param in self.lstm.named_parameters() if 'weight_ih' in name]
        # weight_hh_l = [param.clone() for name, param in self.lstm.named_parameters() if 'weight_hh' in name]
        # bias_ih_l = [param.clone() for name, param in self.lstm.named_parameters() if 'bias_ih' in name]
        # bias_hh_l = [param.clone() for name, param in self.lstm.named_parameters() if 'bias_hh' in name]

        # # LSTM computation step by step
        # all_hidden_states = []
        # hx = h_0
        # cx = c_0
        # for t in range(x.size(0)):  # iterate over time steps
        #     layer_hidden_states = []
        #     for layer in range(self.num_layers):
        #         # Input-to-hidden
        #         gates = torch.mm(x[t] if layer == 0 else all_hidden_states[-1][layer][0], weight_ih_l[layer].t()) + bias_ih_l[layer]
                
        #         # Hidden-to-hidden
        #         gates += torch.mm(hx[layer], weight_hh_l[layer].t()) + bias_hh_l[layer]
                
        #         # Gates are i, f, g, o
        #         i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
                
        #         i_gate = torch.sigmoid(i_gate)
        #         f_gate = torch.sigmoid(f_gate)
        #         g_gate = torch.tanh(g_gate)
        #         o_gate = torch.sigmoid(o_gate)
                
        #         cx[layer] = f_gate * cx[layer] + i_gate * g_gate
        #         hx[layer] = o_gate * torch.tanh(cx[layer])
                
        #         layer_hidden_states.append((hx[layer], cx[layer]))
            
        #     all_hidden_states.append(layer_hidden_states)
        
        # # Collect final hidden states for each time step
        # y = torch.stack([all_hidden_states[t][-1][0] for t in range(x.size(0))], dim=0)
        
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
