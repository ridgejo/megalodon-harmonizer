import torch
import torch.nn as nn
import torch.nn.functional as F

class VADClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VADClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

    def forward(self, x, labels):
        z = self.model(x).squeeze(-1)

        bce_loss = F.binary_cross_entropy_with_logits(x, labels)

        preds = torch.round(F.sigmoid(z))

        balacc = TM.classification.accuracy(
            preds.int(),
            labels.int(),
            task="multiclass",
            num_classes=2,
            average="macro",
        )

        return {
            "vad_bce_loss": bce_loss,
            "vad_balacc": balacc,
        }