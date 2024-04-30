import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TM


class VoicedClassifierLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(VoicedClassifierLSTM, self).__init__()

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

        probs = F.sigmoid(z)
        preds = torch.round(probs)

        balacc = TM.classification.accuracy(
            preds.int(),
            labels.int(),
            task="multiclass",
            num_classes=2,
            average="macro",
        )

        # Computes tensor of shape (5,) of 
        # (true positives, false positives, true negatives, false negatives, support)
        stat_scores = TM.classification.binary_stat_scores(
            preds.int(),
            labels.int(),
        )
        tp = stat_scores[0]
        fp = stat_scores[1]
        tn = stat_scores[2]
        fn = stat_scores[3]
        support = stat_scores[4]

        r2_score = TM.r2_score(
            preds=probs,
            target=labels,
        )

        return {
            "voiced_bce_loss": bce_loss,
            "voiced_balacc": balacc,
            "voiced_r2": r2_score,
            "voiced_tp": tp,
            "voiced_fp": fp,
            "voiced_tn": tn,
            "voiced_fn": fn,
            "voiced_support": support,
        }


class VoicedClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VoicedClassifierMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=1,
            ),
        )

    def forward(self, x, labels):
        x = x.flatten(start_dim=1, end_dim=-1)  # [B, T, E] -> [B, T * E]
        z = self.model(x).squeeze(-1)

        bce_loss = F.binary_cross_entropy_with_logits(z, labels)

        probs = F.sigmoid(z)
        preds = torch.round(probs)

        balacc = TM.classification.accuracy(
            preds.int(),
            labels.int(),
            task="multiclass",
            num_classes=2,
            average="macro",
        )

        r2_score = TM.r2_score(
            preds=probs,
            target=labels,
        )

        return {
            "voiced_bce_loss": bce_loss,
            "voiced_balacc": balacc,
            "voiced_r2": r2_score,
        }
