import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock


def _make_mlp_seq(
    dataset_sizes,
    use_data_block,
    subject_ids,
    use_sub_block,
    feature_dim,
    hidden_dim,
    output_classes,
):

    dataset_layer = DatasetLayer(
        dataset_sizes=dataset_sizes,
        shared_dim=feature_dim,
        use_data_block=use_data_block,
    )

    subject_block = SubjectBlock(
        subject_ids=subject_ids,
        in_channels=feature_dim,
        out_channels=feature_dim,
        use_sub_block=use_sub_block,
    )

    return MLPSeq(
        dataset_layer,
        subject_block,
        feature_dim,
        hidden_dim,
        output_classes,
    )


class MLPSeq(nn.Module):
    """Real-time forward-direction LSTM that labels all embeddings in sequence."""

    def __init__(
        self,
        dataset_layer,
        subject_block,
        feature_dim,
        hidden_dim,
        output_classes,
    ):
        super(MLPSeq, self).__init__()

        assert output_classes >= 2

        self.output_classes = output_classes

        self.dataset_layer = dataset_layer
        self.subject_block = subject_block
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1 if output_classes == 2 else output_classes)
        self.inter_act = nn.ReLU()
        self.act = nn.Sigmoid() if output_classes == 2 else nn.Softmax()

    def forward(self, x, labels, dataset_id, subject_id):
        x = self.dataset_layer(x, dataset_id)
        x = self.subject_block(x, subject_id)
        x = x.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]

        B, T, C = x.shape
        x = self.fc(x.flatten(start_dim=0, end_dim=1)) # (B * L) * classes
        x = self.inter_act(x)
        logits = self.fc2(x)

        preds = self.act(logits) # [B * L, 1]

        logits = logits.squeeze()
        preds = preds.squeeze()
        labels = labels.flatten() # [B * L]

        if self.output_classes == 2:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )
            preds = torch.round(preds)
            acc = balanced_accuracy_score(labels.cpu(), preds.detach().cpu())
            f1 = f1_score(labels.cpu(), preds.detach().cpu())
        else:
            loss = F.cross_entropy(
                logits,
                labels,
            )
            preds = torch.argmax(preds, dim=1)
            acc = balanced_accuracy_score(labels.cpu(), preds.detach().cpu())
            f1 = f1_score(labels.cpu(), preds.detach().cpu())

        loss = {
            "loss": loss,
            "balanced_accuracy": acc,
            "f1_score": f1,
            f"D_{dataset_id}": 1,
            f"D_{dataset_id}_loss": loss,
            f"D_{dataset_id}_balanced_accuracy": acc,
            f"D_{dataset_id}_f1_score": f1,
            f"S_{dataset_id}_{subject_id}": 1,
            f"S_{dataset_id}_{subject_id}_loss": loss,
            f"S_{dataset_id}_{subject_id}_balanced_accuracy": acc,
            f"S_{dataset_id}_{subject_id}_f1_score": f1,
        }

        return preds, loss
