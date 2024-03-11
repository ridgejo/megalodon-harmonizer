import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TM

from dataloaders.multi_dataloader import get_key_from_batch_identifier
from models.brain_encoders.seanet.seanet import SEANetBrainEncoder
from models.dataset_block import DatasetBlock
from models.subject_embedding import SubjectEmbedding


def make_phase_amp_regressor(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim),
        nn.ReLU(),
        nn.Linear(in_features=hidden_dim, out_features=2),
    )


def make_argmax_amp_predictor(input_dim, hidden_dim, dataset_keys):
    class ArgmaxAmpPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, dataset_keys, target_divisor=300):
            super(ArgmaxAmpPredictor, self).__init__()

            self.target_divisor = target_divisor

            self.body = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
            )

            self.output_layer = nn.ModuleDict(
                {
                    dataset_key: nn.Linear(in_features=hidden_dim, out_features=1)
                    for dataset_key in dataset_keys
                }
            )

        def forward(self, x, dataset_key, targets):
            z = self.body(x)
            z = self.output_layer[dataset_key](z)

            # This is much closer to a regression than a classification problem.
            # Scale targets down to approximately [0, 1] range for stability.
            mse_loss = F.mse_loss(z.squeeze(-1), targets.float() / self.target_divisor)

            # Compute real distance to sensor prediction
            rmse_metric = torch.sqrt(
                F.mse_loss(z.squeeze(-1) * self.target_divisor, targets.float())
            )

            return mse_loss, rmse_metric

    return ArgmaxAmpPredictor(input_dim, hidden_dim, dataset_keys)


def make_vad_classifier(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim),
        nn.ReLU(),
        nn.Linear(in_features=hidden_dim, out_features=1),
    )


class RepLearner(L.LightningModule):
    """
    Representation learner.
    """

    def __init__(self, rep_config, batch_size):
        super().__init__()

        # Requirements:
        # - Spatial (attention?) layer
        # - Transformer block
        # - Dataset-conditioned blocks
        # - Subject-conditioned embeddings (16 to 32-dimensional vectors learned in the network)
        # - VQVAE usage or inclusion, and hyperparameters
        # - (SEANet?) Encoder specification
        # - Auxiliary: decoder spec, classifier specs, pre-defined task specs, spectral losses, etc.

        # Predict index of highest amplitude channel? -> needs dataset-conditioned output layer.
        # Circular shift channel and predict index?
        # etc.

        self.lr = rep_config["lr"]
        self.batch_size = batch_size

        active_models = {}

        if "dataset_block" in rep_config:
            active_models["dataset_block"] = DatasetBlock(**rep_config["dataset_block"])

        if "encoder" in rep_config:
            active_models["encoder"] = SEANetBrainEncoder(**rep_config["encoder"])

        if "transformer" in rep_config:
            active_models["transformer"] = nn.TransformerEncoder(
                **rep_config["transformer"]
            )

        # todo: Subject embeddings only in the classifier stage so we don't need to retrain encoder for novel subjects
        # How to implement subject embedding?
        if "subject_embedding" in rep_config:
            subject_embedding_dim = rep_config["subject_embedding"]["embedding_dim"]
            active_models["subject_embedding"] = SubjectEmbedding(
                **rep_config["subject_embedding"]
            )

        if "phase_amp_regressor" in rep_config:
            if "subject_embedding" in rep_config:
                rep_config["phase_amp_regressor"]["input_dim"] += subject_embedding_dim
            active_models["phase_amp_regressor"] = make_phase_amp_regressor(
                **rep_config["phase_amp_regressor"]
            )

        if "argmax_amp_predictor" in rep_config:
            if "subject_embedding" in rep_config:
                rep_config["argmax_amp_predictor"]["input_dim"] += subject_embedding_dim
            active_models["argmax_amp_predictor"] = make_argmax_amp_predictor(
                **rep_config["argmax_amp_predictor"]
            )

        if "vad_classifier" in rep_config:
            if "subject_embedding" in rep_config:
                rep_config["vad_classifier"]["input_dim"] += subject_embedding_dim
            active_models["vad_classifier"] = make_vad_classifier(
                **rep_config["vad_classifier"]
            )

        self.active_models = nn.ModuleDict(active_models)
        self.rep_config = rep_config

    def forward(self, inputs):
        x = inputs["data"]
        z = x.clone()  # Operate on a copy

        dataset = inputs["identifier"]["dataset"][0]
        subject = inputs["identifier"]["subject"][0]

        if "dataset_block" in self.active_models:
            z = self.active_models["dataset_block"](z, dataset_id=dataset)

        if "encoder" in self.active_models:
            z = self.active_models["encoder"](z)

        if "transformer" in self.active_models:
            z = self.active_models["transformer"](z)

        # todo: vector quantization

        # Create two different views for sequence models and independent classifiers
        z_sequence = z.permute(0, 2, 1)  # [B, T, E]
        z_independent = z_sequence.flatten(start_dim=0, end_dim=1)  # [B * T, E]

        if "subject_embedding" in self.active_models:
            subject_embedding = self.active_models["subject_embedding"](
                dataset, subject
            )
            z_ind_subject_embedding = subject_embedding.unsqueeze(0).repeat(
                z_independent.shape[0], 1
            )
            z_seq_subject_embedding = (
                subject_embedding.unsqueeze(0)
                .unsqueeze(1)
                .repeat(z_sequence.shape[0], z_sequence.shape[1], 1)
            )
            z_independent = torch.cat(
                (z_independent, z_ind_subject_embedding), dim=-1
            )  # [B * T, E + S]
            z_sequence = torch.cat(
                (z_sequence, z_seq_subject_embedding), dim=-1
            )  # [B, T, E + S]

        return_values = {}

        if "phase_amp_regressor" in self.active_models:
            pa = self.active_models["phase_amp_regressor"](z_independent)
            phase, amp = pa[:, 0], pa[:, 1]
            return_values["phase"] = phase
            return_values["amp"] = amp

        if "argmax_amp_predictor" in self.active_models:
            argmax_amp = self.active_models["argmax_amp_predictor"](
                z_independent,
                dataset_key=dataset,
                targets=self.compute_argmax_amp(x),
            )
            return_values["argmax_amp"] = argmax_amp

        if "vad_classifier" in self.active_models and "vad_labels" in inputs:
            vad_logits = self.active_models["vad_classifier"](z_independent).squeeze(-1)
            return_values["vad_logits"] = vad_logits

        return return_values

    def training_step(self, batch, batch_idx):
        loss, losses, metrics = self._shared_step(batch, batch_idx, "train")

        if loss is not None:
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, losses, metrics = self._shared_step(batch, batch_idx, "val")

        if loss is not None:
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, losses, metrics = self._shared_step(batch, batch_idx, "test")

        if loss is not None:
            self.log(
                "test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
            )

        return loss

    def _shared_step(self, batch, batch_idx, stage: str):
        loss = 0.0
        losses = {}
        metrics = {}

        data_key = get_key_from_batch_identifier(batch["identifier"])
        dataset = batch["identifier"]["dataset"][0]

        return_values = self(batch)

        # Compute losses over this data batch
        for key, val in return_values.items():
            if key == "argmax_amp":
                # Compute max amplitude index at all time points in signal

                argmax_amp_loss, rmse_metric = val

                loss += argmax_amp_loss
                losses[f"{stage}+argmax_amp_loss"]
                losses[f"{stage}_{data_key}+argmax_amp_loss"] = argmax_amp_loss
                losses[f"{stage}_dat={dataset}+argmax_amp_loss"] = argmax_amp_loss

                metrics[f"{stage}_{data_key}+argmax_amp_rmse"] = rmse_metric
                metrics[f"{stage}_dat={dataset}+argmax_amp_rmse"] = rmse_metric

            if key == "vad_logits":
                vad_logits = val
                vad_labels = batch["vad_labels"].flatten(start_dim=0, end_dim=1)
                vad_loss = F.binary_cross_entropy_with_logits(vad_logits, vad_labels)

                vad_preds = torch.round(F.sigmoid(vad_logits))
                vad_balacc = TM.classification.accuracy(
                    vad_preds.int(),
                    vad_labels.int(),
                    task="multiclass",
                    num_classes=2,
                    average="macro",
                )

                loss += vad_loss
                losses[f"{stage}+vad_bce_loss"] = vad_loss
                losses[f"{stage}_{data_key}+vad_bce_loss"] = vad_loss
                losses[f"{stage}_dat={dataset}+vad_bce_loss"] = vad_loss

                metrics[f"{stage}_{data_key}+vad_balacc"] = vad_balacc
                metrics[f"{stage}_dat={dataset}+vad_balacc"] = vad_balacc

        return loss, losses, metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.active_models.parameters(), lr=self.lr)

    def compute_phase_amp(self, x):
        X_fft = torch.fft.fft(x)
        amp = torch.abs(X_fft)
        phase = torch.angle(X_fft)
        return (phase, amp)

    def compute_argmax_amp(self, x):
        amplitude = torch.abs(x)
        argmax = torch.argmax(amplitude, dim=1)
        return argmax.flatten(start_dim=0, end_dim=1)


if __name__ == "__main__":
    model = RepLearner(
        rep_config={
            "lr": 0.001,
            "dataset_block": {
                "dataset_sizes": {
                    "armeni2022": 269,
                    "schoffelen2019": 273,
                    "gwilliams2022": 208,
                },
                "shared_dim": 300,
                "use_data_block": True,
            },
            "encoder": {
                "channels": 300,
                "conv_channels": [512, 512],
                "ratios": [1],
                "dimension": 256,
            },
            "argmax_amp_predictor": {
                "input_dim": 256,
                "hidden_dim": 512,
                "dataset_keys": ["armeni2022", "schoffelen2019", "gwilliams2022"],
            },
            # "phase_amp_regressor": {
            #     "input_dim": 256,
            #     "hidden_dim": 512,
            # },
        },
        batch_size=32,
    )

    def test_with_mocked_data():
        x = torch.randn(32, 269, 750)
        model.training_step(
            {
                "data": x,
                "labels": None,
                "times": None,
            },
            0,
        )

    def test_with_real_data():
        from dataloaders.multi_dataloader import MultiDataLoader

        datamodule = MultiDataLoader(
            dataset_preproc_configs={
                "armeni2022": {
                    "bad_subjects": [],
                    "bad_sessions": {"001": [], "002": [], "003": []},
                    "slice_len": 0.1,
                    "label_type": None,
                },
                "schoffelen2019": {
                    "bad_subjects": [],
                    "slice_len": 0.1,
                    "label_type": None,
                },
                "gwilliams2022": {
                    "bad_subjects": [],
                    "slice_len": 0.1,
                    "label_type": "vad",
                },
            },
            dataloader_configs={
                "train_ratio": 0.9,
                "val_ratio": 0.04,
                "test_ratio": 0.04,
                "pred_ratio": 0.02,
                "batch_size": 32,
                "normalisation": {
                    "n_sample_batches": 8,
                    "per_channel": True,
                    "scaler_conf": {"standard_scaler": None},
                },
            },
            debug=True,
        )

        trainer = L.pytorch.trainer.trainer.Trainer()
        trainer.fit(model, datamodule=datamodule)

    test_with_real_data()
