import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders.multi_dataloader import get_key_from_batch_identifier
from models.brain_encoders.amp_ssl.amp_scale_predictor import AmpScalePredictor
from models.brain_encoders.freq_ssl.band_predictor import BandPredictor
from models.brain_encoders.phase_ssl.phase_diff_predictor import PhaseDiffPredictor
from models.brain_encoders.seanet.seanet import SEANetBrainEncoder
from models.brain_encoders.spatial_ssl.masked_channel_predictor import (
    MaskedChannelPredictor,
)
from models.brain_encoders.supervised.vad_classifier import VADClassifier
from models.brain_encoders.supervised.voiced_classifier import (
    VoicedClassifierLSTM,
    VoicedClassifierMLP,
)
from models.dataset_block import DatasetBlock
from models.subject_block import SubjectBlock
from models.subject_embedding import SubjectEmbedding
from models.transformer_encoder import TransformerEncoder
from models.vector_quantize import VectorQuantize

class RepLearner(L.LightningModule):
    """
    Representation learner.
    """

    def __init__(self, rep_config):
        super().__init__()

        # Requirements:
        # - Spatial (attention?) layer (requires sensor geometry) --- Bin into 50 (?)
        # - Auxiliary: decoder spec, classifier specs, pre-defined task specs, spectral losses, etc.
        # Circular shift channel and predict index?
        # etc.

        self.learning_rate = rep_config["lr"]
        self.weightings = {}

        active_models = {}

        if "dataset_block" in rep_config:
            active_models["dataset_block"] = DatasetBlock(**rep_config["dataset_block"])

        if "encoder" in rep_config:
            active_models["encoder"] = SEANetBrainEncoder(**rep_config["encoder"])

        if "transformer" in rep_config:
            active_models["transformer"] = TransformerEncoder(
                in_channels=rep_config["encoder"]["dimension"],
                transformer_config=rep_config["transformer"],
            )

        if "quantize" in rep_config:
            self.weightings["quantization"] = rep_config["quantize"].get("weight", 1.0)
            rep_config["quantize"].pop("weight", None)
            active_models["quantize"] = VectorQuantize(rep_config["quantize"])
        else:
            self.weightings["quantization"] = 0.0

        if "subject_embedding" in rep_config:
            subject_embedding_dim = rep_config["subject_embedding"]["embedding_dim"]
            active_models["subject_embedding"] = SubjectEmbedding(
                **rep_config["subject_embedding"]
            )
        elif "subject_block" in rep_config:
            active_models["subject_block"] = SubjectBlock(**rep_config["subject_block"])

        # Auxiliary SSL losses
        if "masked_channel_predictor" in rep_config:
            self.weightings["masked_channel_pred"] = rep_config[
                "masked_channel_predictor"
            ].get("weight", 1.0)
            rep_config["masked_channel_predictor"].pop("weight", None)

            if "subject_embedding" in rep_config:
                rep_config["masked_channel_predictor"][
                    "input_dim"
                ] += subject_embedding_dim
            active_models["masked_channel_predictor"] = MaskedChannelPredictor(
                **rep_config["masked_channel_predictor"]
            )

        if "band_predictor" in rep_config:
            self.weightings["band_predictor"] = rep_config["band_predictor"].get(
                "weight", 1.0
            )
            rep_config["band_predictor"].pop("weight", None)

            if "subject_embedding" in rep_config:
                rep_config["band_predictor"]["input_dim"] += subject_embedding_dim
            active_models["band_predictor"] = BandPredictor(
                **rep_config["band_predictor"]
            )

        if "phase_diff_predictor" in rep_config:
            self.weightings["phase_diff_predictor"] = rep_config[
                "phase_diff_predictor"
            ].get("weight", 1.0)

            if "subject_embedding" in rep_config:
                rep_config["phase_diff_predictor"]["input_dim"] += subject_embedding_dim

            active_models["phase_diff_predictor"] = PhaseDiffPredictor(
                **rep_config["phase_diff_predictor"]
            )

        if "amp_scale_predictor" in rep_config:
            self.weightings["amp_scale_predictor"] = rep_config[
                "amp_scale_predictor"
            ].get("weight", 1.0)

            if "subject_embedding" in rep_config:
                rep_config["amp_scale_predictor"]["input_dim"] += subject_embedding_dim

            active_models["amp_scale_predictor"] = AmpScalePredictor(
                **rep_config["amp_scale_predictor"]
            )

        # Label losses for representation shaping
        if "vad_classifier" in rep_config:
            self.weightings["vad"] = rep_config["vad_classifier"].get("weight", 1.0)
            rep_config["vad_classifier"].pop("weight", None)

            if "subject_embedding" in rep_config:
                rep_config["vad_classifier"]["input_dim"] += subject_embedding_dim
            active_models["vad_classifier"] = VADClassifier(
                **rep_config["vad_classifier"]
            )
        if "voiced_classifier" in rep_config:
            self.weightings["voiced"] = rep_config["voiced_classifier"].get(
                "weight", 1.0
            )
            rep_config["voiced_classifier"].pop("weight", None)

            if "subject_embedding" in rep_config:
                rep_config["voiced_classifier"]["input_dim"] += subject_embedding_dim

            if rep_config["voiced_classifier"]["type"] == "mlp":
                del rep_config["voiced_classifier"]["type"]
                active_models["voiced_classifier"] = VoicedClassifierMLP(
                    **rep_config["voiced_classifier"]
                )
            elif rep_config["voiced_classifier"]["type"] == "lstm":
                del rep_config["voiced_classifier"]["type"]
                active_models["voiced_classifier"] = VoicedClassifierLSTM(
                    **rep_config["voiced_classifier"]
                )
            else:
                raise ValueError("Voiced classifier type not recognised")

        self.active_models = nn.ModuleDict(active_models)
        self.rep_config = rep_config

    def apply_encoder(self, z, dataset, subject):
        if "dataset_block" in self.active_models:
            z = self.active_models["dataset_block"](z, dataset_id=dataset)

        if "encoder" in self.active_models:
            z = self.active_models["encoder"](z)

        if "transformer" in self.active_models:
            z = self.active_models["transformer"](z)

        if "quantize" in self.active_models:
            z, _, commit_loss = self.active_models["quantize"](z)
        else:
            commit_loss = 0.0

        if "subject_block" in self.active_models:
            z = self.active_models["subject_block"](z, dataset, subject)

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

        return z_sequence, z_independent, commit_loss

    def forward(self, inputs):
        x = inputs["data"]
        sensor_pos = inputs["sensor_pos"]

        dataset = inputs["identifier"]["dataset"][0]
        subject = inputs["identifier"]["subject"][0]

        z_sequence, z_independent, commit_loss = self.apply_encoder(x, dataset, subject)

        return_values = {"quantization": {"commit_loss": commit_loss}}

        if "band_predictor" in self.active_models:
            x_filtered, band_label = self.active_models["band_predictor"].filter_band(
                x, sample_rate=250
            )  # warning: hardcoded
            z_filtered_sequence, _, _ = self.apply_encoder(x_filtered, dataset, subject)
            return_values["band_predictor"] = self.active_models["band_predictor"](
                z_filtered_sequence, band_label
            )

        if "phase_diff_predictor" in self.active_models:
            x_shifted, phase_label = self.active_models[
                "phase_diff_predictor"
            ].apply_random_phase_shift(x)
            z_shifted_sequence, _, _ = self.apply_encoder(x_shifted, dataset, subject)
            return_values["phase_diff_predictor"] = self.active_models[
                "phase_diff_predictor"
            ](z_shifted_sequence, phase_label)

        if "masked_channel_predictor" in self.active_models:
            x_masked, mask_label = self.active_models[
                "masked_channel_predictor"
            ].mask_input(x, sensor_pos)
            # todo: do something with commit loss
            z_mask_sequence, _, _ = self.apply_encoder(x_masked, dataset, subject)
            return_values["masked_channel_pred"] = self.active_models[
                "masked_channel_predictor"
            ](z_mask_sequence, mask_label)

        if "amp_scale_predictor" in self.active_models:
            x_scaled, scale_label = self.active_models["amp_scale_predictor"].scale_amp(
                x
            )
            z_scaled_sequence, _, _ = self.apply_encoder(x_scaled, dataset, subject)
            return_values["amp_scale_predictor"] = self.active_models[
                "amp_scale_predictor"
            ](z_scaled_sequence, scale_label)

        if "vad_classifier" in self.active_models and "vad_labels" in inputs:
            vad_labels = inputs["vad_labels"]  # [B, T]

            if vad_labels.shape[-1] != z_sequence.shape[1]:
                # Downsample labels to match number of encoder output embeddings
                vad_labels = F.interpolate(
                    vad_labels.unsqueeze(1),  # [B, 1, T]
                    size=z_sequence.shape[1],  # T2
                ).squeeze(1)  # [B, T2]

            return_values["vad"] = self.active_models["vad_classifier"](
                z_independent, vad_labels.flatten(start_dim=0, end_dim=-1)
            )

        if "voiced_classifier" in self.active_models and "voiced_labels" in inputs:
            voiced_labels = inputs["voiced_labels"]
            return_values["voiced"] = self.active_models["voiced_classifier"](
                z_sequence, voiced_labels
            )

        return return_values

    def training_step(self, batch, batch_idx):
        loss, losses, metrics = self._shared_step(batch, batch_idx, "train")

        batch_size = len(batch["data"])

        if loss is not None:
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, losses, metrics = self._shared_step(batch, batch_idx, "val")

        batch_size = len(batch["data"])

        if loss is not None:
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, losses, metrics = self._shared_step(batch, batch_idx, "test")

        batch_size = len(batch["data"])

        if loss is not None:
            self.log(
                "test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
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
            loss_weighting = self.weightings[key]

            # Each return value from a module contains a dictionary of losses and metrics
            for k, v in val.items():
                if "loss" in k:
                    v = v * loss_weighting
                    loss += v
                    losses[f"{stage}+{k}"] = v
                    losses[f"{stage}_{data_key}+{k}"] = v
                    losses[f"{stage}_dat={dataset}+{k}"] = v
                else:
                    metrics[f"{stage}+{k}"] = v
                    metrics[f"{stage}_{data_key}+{k}"] = v
                    metrics[f"{stage}_dat={dataset}+{k}"] = v

        return loss, losses, metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                self.active_models.parameters(),  # Prevents unintentional unfreezing
            ),
            lr=self.learning_rate,
        )

    def freeze_except(self, module_name: str):
        for key in self.active_models.keys():
            if module_name not in key:
                for param in self.active_models[key].parameters():
                    param.requires_grad = False


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
