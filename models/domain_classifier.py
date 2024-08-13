# Nicola Dinsdale 2020, Unlearning_for_MRI_harmonisation
# Model for unlearning domain

import torch.nn as nn
import torch
from collections import OrderedDict

class DomainClassifier(nn.Module): #maybe try linear SVM 
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
    
class LSVM_DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSVM_DomainClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # No activation function like ReLU or Softmax

#TODO remove
class LeakyDomainClassifier(nn.Module):
    def __init__(self, nodes=2, init_features=2560, batch_size=512):
        super(LeakyDomainClassifier, self).__init__()
        self.nodes = nodes
        self.domain = nn.Sequential()
        # 512 is the size output by the final layer of the encoder
        self.domain.add_module('d_fc2', nn.Linear(init_features, batch_size)) #TODO investigate whether this is too big of a reduction
        self.domain.add_module('d_relu2', nn.LeakyReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout(p=0.2)) # changed from Dropout3D but should investigate whether features has the right dimensionality
        self.domain.add_module('d_fc3', nn.Linear(batch_size, nodes))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        domain_pred = self.domain(x)
        return domain_pred
