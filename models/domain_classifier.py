# Nicola Dinsdale 2020, Unlearning_for_MRI_harmonisation
# Model for unlearning domain

import torch.nn as nn
from collections import OrderedDict

class DomainClassifier(nn.Module):
    def __init__(self, nodes=2, init_features=2560, batch_size=512):
        super(DomainClassifier, self).__init__()
        self.nodes = nodes
        self.domain = nn.Sequential()
        # 512 is the size output by the final layer of the encoder
        self.domain.add_module('d_fc2', nn.Linear(init_features, batch_size)) #TODO investigate whether this is too big of a reduction
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout(p=0.2)) # changed from Dropout3D but should investigate whether features has the right dimensionality
        self.domain.add_module('d_fc3', nn.Linear(batch_size, nodes))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        domain_pred = self.domain(x)
        return domain_pred


class DomainPredictor(nn.Module):
    def __init__(self, n_domains=2, init_features=64):
        super(DomainPredictor, self).__init__()
        self.n_domains = n_domains
        features = init_features

        self.decoder1 = DomainPredictor._half_block(features, features, name="conv1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder2 = DomainPredictor._half_block(features, features, name="conv2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder3 = DomainPredictor._half_block(features, features, name="conv3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder4 = DomainPredictor._half_block(features, features, name="conv3")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder5 = DomainPredictor._projector_block(features, 1, name="projectorblock")
        # Projector block to reduce features
        self.domain = nn.Sequential()
        self.domain.add_module('r_fc1', nn.Linear(121, 96))
        self.domain.add_module('r_relu1', nn.ReLU(True)) #TODO use leaky ReLU if vanishing gradients persists
        self.domain.add_module('d_fc2', nn.Linear(96, 32))
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout2d(p=0.2))
        self.domain.add_module('d_fc3', nn.Linear(32, n_domains))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        dec1 = self.decoder1(x)
        dec2 = self.decoder2(self.pool1(dec1))
        dec3 = self.decoder3(self.pool2(dec2))
        dec4 = self.decoder4(self.pool3(dec3))
        dec5 = self.decoder5(self.pool3(dec4))

        dec5 = dec5.view(-1, 121)
        domain_pred = self.domain(dec5)
        return domain_pred

    @staticmethod
    def _projector_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _half_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )