import lightning as L
import torch
import typing as tp
import torch.nn as nn
import torch.nn.functional as F

from models.brain_encoders.seanet.seanet import SEANetBrainEncoder, SEANetBrainDecoder
from models.dataset_block import DatasetBlock

def get_key_from_identifier(key: str, identifier: str):
    """Get key from identifier in style dat=..._sub=..._ses=..."""
    return identifier.split(f"{key}=")[1].split("_")[0]

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

            self.output_layer = nn.ModuleDict({
                dataset_key: nn.Linear(
                        in_features=hidden_dim, out_features=1
                    ) for dataset_key in dataset_keys
            })

        def forward(self, x, dataset_key, targets):
            z = self.body(x)
            z = self.output_layer[dataset_key](z)

            # This is much closer to a regression than a classification problem.
            # Scale targets down to approximately [0, 1] range for stability.
            mse_loss = F.mse_loss(z.squeeze(-1), targets.float() / self.target_divisor)

            # Compute real distance to sensor prediction
            rmse_metric = torch.sqrt(F.mse_loss(z.squeeze(-1) * self.target_divisor, targets.float()))

            return mse_loss, rmse_metric

    return ArgmaxAmpPredictor(input_dim, hidden_dim, dataset_keys)

class RepLearner(L.LightningModule):
    """
    Representation learner.
    """

    def __init__(self, rep_config):
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

        active_models = {}

        if "dataset_block" in rep_config:
            active_models["dataset_block"] = DatasetBlock(
                **rep_config["dataset_block"]
            )
        
        active_models["encoder"] = SEANetBrainEncoder(
            **rep_config["encoder"]
        )

        if "transformer" in rep_config:
            active_models["transformer"] = nn.TransformerEncoder(
                **rep_config["transformer"]
            )

        # todo: Subject embeddings only in the classifier stage so we don't need to retrain encoder for novel subjects

        if "phase_amp_regressor" in rep_config:
            active_models["phase_amp_regressor"] = make_phase_amp_regressor(
                **rep_config["phase_amp_regressor"]
            )

        if "argmax_amp_predictor" in rep_config:
            active_models["argmax_amp_predictor"] = make_argmax_amp_predictor(
                **rep_config["argmax_amp_predictor"]
            )

        self.active_models = nn.ModuleDict(active_models)
        self.rep_config = rep_config

    def forward(self, inputs, identifier):

        x = inputs["data"]
        z = x.clone() # Operate on a copy

        if "dataset_block" in self.rep_config:
            z = self.active_models["dataset_block"](
                z,
                dataset_id=get_key_from_identifier("dat", identifier)
            )
        
        z = self.active_models["encoder"](z)

        if "transformer" in self.rep_config:
            z = self.active_models["transformer"](z)

        # Batch the temporal dimension for independent classification [B, E, T] -> [B * T, E]
        z = z.permute(0, 2, 1).flatten(start_dim=0, end_dim=1)

        return_values = {}

        if "phase_amp_regressor" in self.rep_config:
            pa = self.active_models["phase_amp_regressor"](z)
            phase, amp = pa[:, 0], pa[:, 1]
            return_values["phase"] = phase
            return_values["amp"] = amp

        if "argmax_amp_predictor" in self.rep_config:
            argmax_amp = self.active_models["argmax_amp_predictor"](
                z,
                dataset_key=get_key_from_identifier("dat", identifier),
                targets=self.compute_argmax_amp(x)
            )
            return_values["argmax_amp"] = argmax_amp

        return return_values

    def training_step(self, batch, batch_idx):

        # batch will contains keys from all datasets.
        # each of these will contain keys for data/labels/times/etc.

        losses = {}
        metrics = {}
        for data_key, data_batch in batch.items():

            if data_batch is not None:

                return_values = self(data_batch, data_key)

                # Compute losses over this data batch
                for key, val in return_values.items():

                    if key == "argmax_amp":
                        # Compute max amplitude index at all time points in signal

                        argmax_amp_loss, rmse_metric = val
                        losses[f"train_{data_key}+argmax_amp_loss"] = argmax_amp_loss
                        metrics[f"train_{data_key}+argmax_amp_rmse"] = rmse_metric
        
        # Combine all losses to get total loss
        loss = 0
        for key, val in losses.items():
            loss += val

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

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
        }
    )

    def test_with_mocked_data():
        x = torch.randn(32, 269, 750)
        model.training_step({
            "data": x,
            "labels": None,
            "times": None,
        }, 0)

    def test_with_real_data():

        from dataloaders.multi_dataloader import MultiDataLoader
        datamodule = MultiDataLoader(
            dataset_preproc_configs={
                "armeni2022": {
                    "bad_subjects": [],
                    "bad_sessions": {"001": [], "002": [], "003": []},
                    "slice_len": 0.1,
                    "label_type": "vad",
                },
                "schoffelen2019": {
                    "bad_subjects": [],
                    "slice_len": 0.1,
                    "label_type": None,
                },
                "gwilliams2022": {
                    "bad_subjects": [],
                    "slice_len": 0.1,
                    "label_type": None,
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
                    "scaler_conf": {
                        "standard_scaler": None
                    },
                },
            },
            debug=True,
        )

        trainer = L.pytorch.trainer.trainer.Trainer()
        trainer.fit(model, datamodule=datamodule)

    test_with_real_data()