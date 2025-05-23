import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TM

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

        bce_loss = F.binary_cross_entropy_with_logits(z, labels)

        preds = torch.round(F.sigmoid(z))

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
        ).float() / len(preds.flatten())
        tp = stat_scores[0]
        fp = stat_scores[1]
        tn = stat_scores[2]
        fn = stat_scores[3]
        support = stat_scores[4]

        # Track label distribution (as it could be affected by downsampling)
        no_speech_labels = torch.count_nonzero(labels)
        total_labels = torch.prod(torch.tensor(labels.shape)).item()
        pct_speech = (total_labels - no_speech_labels) / total_labels

        return {
            "vad_bce_loss": bce_loss,
            "vad_balacc": balacc,
            "vad_pct_speech_labels": pct_speech,
            "vad_tp": tp,
            "vad_fp": fp,
            "vad_tn": tn,
            "vad_fn": fn,
            "vad_support": support,
        }

class VADClassifierLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VADClassifierLinear, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1),
        )

    def forward(self, x, labels):
        z = self.model(x).squeeze(-1)

        bce_loss = F.binary_cross_entropy_with_logits(z, labels)

        preds = torch.round(F.sigmoid(z))

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
        ).float() / len(preds.flatten())
        tp = stat_scores[0]
        fp = stat_scores[1]
        tn = stat_scores[2]
        fn = stat_scores[3]
        support = stat_scores[4]

        # Track label distribution (as it could be affected by downsampling)
        no_speech_labels = torch.count_nonzero(labels)
        total_labels = torch.prod(torch.tensor(labels.shape)).item()
        pct_speech = (total_labels - no_speech_labels) / total_labels

        return {
            "vad_bce_loss": bce_loss,
            "vad_balacc": balacc,
            "vad_pct_speech_labels": pct_speech,
            "vad_tp": tp,
            "vad_fp": fp,
            "vad_tn": tn,
            "vad_fn": fn,
            "vad_support": support,
        }