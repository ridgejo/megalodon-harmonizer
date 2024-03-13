import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TM


class VoicedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(VoicedClassifier, self).__init__()

        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            in_features=hidden_dim,
            out_features=1,
        )

    def forward(self, x, labels):
        out, (_, _) = self.model(x)
        z = self.classifier(out[:, -1, :]).squeeze(-1)

        bce_loss = F.binary_cross_entropy_with_logits(z, labels)
        preds = torch.round(F.sigmoid(z)).squeeze(-1)

        balacc = TM.classification.accuracy(
            preds.int(),
            labels.int(),
            task="multiclass",
            num_classes=2,
            average="macro",
        )

        return {
            "voiced_bce_loss": bce_loss,
            "voiced_balacc": balacc,
        }