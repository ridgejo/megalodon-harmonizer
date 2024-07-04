# Nicola Dinsdale 2020, Unlearning_for_MRI_harmonisation
# Model for unlearning domain

import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, nodes=2):
        super(DomainClassifier, self).__init__()
        self.nodes = nodes
        self.domain = nn.Sequential()
        self.domain.add_module('d_fc2', nn.Linear(512, 32)) #TODO investigate whether this is too big of a reduction
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout3d(p=0.2))
        self.domain.add_module('d_fc3', nn.Linear(32, nodes))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        domain_pred = self.domain(x)
        return domain_pred