import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

from dataloaders.data_utils import get_key_from_batch_identifier, get_dset_encoding
from models.attach_subject import AttachSubject
from models.brain_encoders.amp_ssl.amp_scale_predictor import AmpScalePredictor
from models.brain_encoders.freq_ssl.band_predictor import BandPredictor
from models.brain_encoders.phase_ssl.phase_diff_predictor import PhaseDiffPredictor
from models.brain_encoders.seanet.seanet import SEANetBrainEncoder
from models.brain_encoders.spatial_ssl.masked_channel_predictor import (
    MaskedChannelPredictor,
)
from models.brain_encoders.supervised.vad_classifier import VADClassifier, VADClassifierLinear
from models.brain_encoders.supervised.voiced_classifier import (
    VoicedClassifierLSTM,
    VoicedClassifierMLP,
    VoicedClassifierLinear,
)
from models.dataset_block import DatasetBlock
from models.film import FiLM
from models.projector import Projector
from models.subject_block import SubjectBlock
from models.subject_embedding import SubjectEmbedding
from models.transformer_encoder import TransformerEncoder
from models.vector_quantize import VectorQuantize
from models.domain_classifier import DomainClassifier, DomainPredictor, LeakyDomainClassifier
from models.confusion_loss import ConfusionLoss


class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args):
        return self.func(*args)


class RepLearnerUnlearner(L.LightningModule):
    """
    Representation learner with dataset unlearning.
    """

    def __init__(self, rep_config):
        super().__init__()
        self.automatic_optimization = False
        self.epoch_stage_1 = rep_config["epoch_stage_1"]

        self.learning_rate = rep_config["lr"]
        self.weightings = {}

        encoder_models = {}
        predictor_models = {}

        # ---- Core encoder ----
        if "dataset_block" in rep_config:
            encoder_models["dataset_block"] = DatasetBlock(**rep_config["dataset_block"])
        else:
            encoder_models["dataset_block"] = LambdaModule(lambda z, ds: z)

        if "encoder" in rep_config:
            encoder_models["encoder"] = SEANetBrainEncoder(**rep_config["encoder"])
        else:
            encoder_models["encoder"] = nn.Identity()

        if "transformer" in rep_config:
            encoder_models["transformer"] = TransformerEncoder(
                in_channels=rep_config["encoder"]["dimension"],
                transformer_config=rep_config["transformer"],
            )
        else:
            encoder_models["transformer"] = nn.Identity()

        if "quantize" in rep_config:
            self.weightings["quantization"] = rep_config["quantize"].get("weight", 1.0)
            rep_config["quantize"].pop("weight", None)
            encoder_models["quantize"] = VectorQuantize(rep_config["quantize"])
        else:
            self.weightings["quantization"] = 0.0
            dummy_module = nn.Identity()
            encoder_models["quantize"] = LambdaModule(
                lambda z: (dummy_module(z), None, 0.0)
            )

        # ---- Subject conditioning ----
        assert (
            sum(
                [
                    x in rep_config
                    for x in ["subject_embedding", "subject_block", "subject_film"]
                ]
            )
            <= 1
        ), "Can't have multiple subject conditioning methods"

        if "subject_block" in rep_config:
            encoder_models["subject_block"] = SubjectBlock(**rep_config["subject_block"])
        else:
            dummy_module = nn.Identity()
            encoder_models["subject_block"] = LambdaModule(
                lambda z, ds, sb: dummy_module(z)
            )

        if "subject_embedding" in rep_config:
            subject_embedding_dim = rep_config["subject_embedding"]["embedding_dim"]
            encoder_models["subject_embedding"] = SubjectEmbedding(
                **rep_config["subject_embedding"]
            )
            encoder_models["attach_subject"] = AttachSubject()
        else:
            encoder_models["subject_embedding"] = LambdaModule(lambda ds, sb: None)
            encoder_models["attach_subject"] = LambdaModule(lambda z, sb: z)

        if "subject_film" in rep_config:
            # In this case, we have subject embeddings which condition the film module
            encoder_models["subject_embedding"] = SubjectEmbedding(
                **rep_config["subject_film"]["subject_embedding"]
            )
            encoder_models["subject_film_module"] = FiLM(
                **rep_config["subject_film"]["film_module"]
            )
        else:
            if "subject_embedding" not in encoder_models:
                encoder_models["subject_embedding"] = LambdaModule(lambda ds, sb: None)
            encoder_models["subject_film_module"] = LambdaModule(lambda z, sb: z)

        # ---- SSL Projector ----
        if "projector" in rep_config:
            if "subject_embedding" in rep_config:
                rep_config["projector"]["input_dim"] += subject_embedding_dim
            encoder_models["projector"] = Projector(**rep_config["projector"])
        else:
            encoder_models["projector"] = nn.Identity()

        # ---- Auxiliary SSL losses ----
        if "masked_channel_predictor" in rep_config:
            self.weightings["masked_channel_pred"] = rep_config[
                "masked_channel_predictor"
            ].get("weight", 1.0)
            rep_config["masked_channel_predictor"].pop("weight", None)

            if "subject_embedding" in rep_config:
                rep_config["masked_channel_predictor"][
                    "input_dim"
                ] += subject_embedding_dim
            predictor_models["masked_channel_predictor"] = MaskedChannelPredictor(
                **rep_config["masked_channel_predictor"]
            )

        if "band_predictor" in rep_config:
            self.weightings["band_predictor"] = rep_config["band_predictor"].get(
                "weight", 1.0
            )
            rep_config["band_predictor"].pop("weight", None)

            if "subject_embedding" in rep_config:
                rep_config["band_predictor"]["input_dim"] += subject_embedding_dim
            predictor_models["band_predictor"] = BandPredictor(
                **rep_config["band_predictor"]
            )

        if "phase_diff_predictor" in rep_config:
            self.weightings["phase_diff_predictor"] = rep_config[
                "phase_diff_predictor"
            ].get("weight", 1.0)

            if "subject_embedding" in rep_config:
                rep_config["phase_diff_predictor"]["input_dim"] += subject_embedding_dim

            predictor_models["phase_diff_predictor"] = PhaseDiffPredictor(
                **rep_config["phase_diff_predictor"]
            )

        if "amp_scale_predictor" in rep_config:
            self.weightings["amp_scale_predictor"] = rep_config[
                "amp_scale_predictor"
            ].get("weight", 1.0)

            if "subject_embedding" in rep_config:
                rep_config["amp_scale_predictor"]["input_dim"] += subject_embedding_dim

            predictor_models["amp_scale_predictor"] = AmpScalePredictor(
                **rep_config["amp_scale_predictor"]
            )

        self.encoder_models = nn.ModuleDict(encoder_models)
        self.predictor_models = nn.ModuleDict(predictor_models)
        self.domain_classifier = LeakyDomainClassifier(nodes=2, init_features=2560, batch_size=512) # nodes = number of datasets (I think)
        # self.domain_classifier = DomainPredictor(n_domains=2, init_features=512)
        self.rep_config = rep_config
        self.domain_criterion = nn.CrossEntropyLoss() # nn.BCELoss() to be used with DomainPredictor
        self.conf_criterion = ConfusionLoss()

        # Add classifiers if used in pre-training
        for k, v in rep_config.items():
            if "classifier" in k:
                self.add_classifier(k, v)

    def apply_encoder(self, z, dataset, subject):
        z = self.encoder_models["dataset_block"](z, dataset)
        z = self.encoder_models["encoder"](z)
        z = self.encoder_models["transformer"](z)
        z, _, commit_loss = self.encoder_models["quantize"](z)

        # Generic subject embedding
        subject_embedding = self.encoder_models["subject_embedding"](dataset, subject)

        # Subject block
        z = self.encoder_models["subject_block"](z, dataset, subject)

        # Subject FiLM conditioning
        z = self.encoder_models["subject_film_module"](z, subject_embedding)

        # Subject embedding concatentation
        z = self.encoder_models["attach_subject"](z, subject_embedding)

        # Create two different views for sequence models and independent classifiers
        z_sequence = z.permute(0, 2, 1)  # [B, T, E]
        z_independent = z_sequence.flatten(start_dim=0, end_dim=1)  # [B * T, E]

        # Apply SSL projector to z_sequence
        T, E = z_sequence.shape[1:]
        z_projected = self.encoder_models["projector"](
            z_sequence.flatten(start_dim=1, end_dim=-1)
        )
        z_sequence = torch.unflatten(z_projected, dim=-1, sizes=(T, E))

        # The logits or final features output
        features = z_projected.view(z.size(0), -1)  # [batch_size, _]

        return features, z_sequence, z_independent, commit_loss

    def forward(self, inputs):
        x = inputs["data"]

        # sensor_pos = inputs["sensor_pos"]
        sensor_pos = None

        dataset = inputs["info"]["dataset"][0]
        subject = inputs["info"]["subject_id"]

        features, z_sequence, z_independent, commit_loss = self.apply_encoder(x, dataset, subject)

        return_values = {"quantization": {"commit_loss": commit_loss},
                         "classifier features": features}

        if "band_predictor" in self.predictor_models:
            x_filtered, band_label = self.predictor_models["band_predictor"].filter_band(
                x, sample_rate=250
            )  # warning: hardcoded
            _, z_filtered_sequence, _, _ = self.apply_encoder(x_filtered, dataset, subject)
            return_values["band_predictor"] = self.predictor_models["band_predictor"](
                z_filtered_sequence, band_label
            )

        if "phase_diff_predictor" in self.predictor_models:
            x_shifted, phase_label = self.predictor_models[
                "phase_diff_predictor"
            ].apply_random_phase_shift(x)
            _, z_shifted_sequence, _, _ = self.apply_encoder(x_shifted, dataset, subject)
            return_values["phase_diff_predictor"] = self.predictor_models[
                "phase_diff_predictor"
            ](z_shifted_sequence, phase_label)

        if "masked_channel_predictor" in self.predictor_models:
            x_masked, mask_label = self.predictor_models[
                "masked_channel_predictor"
            ].mask_input(x, sensor_pos)
            # todo: do something with commit loss
            _, z_mask_sequence, _, _ = self.apply_encoder(x_masked, dataset, subject)
            return_values["masked_channel_pred"] = self.predictor_models[
                "masked_channel_predictor"
            ](z_mask_sequence, mask_label)

        if "amp_scale_predictor" in self.predictor_models:
            x_scaled, scale_label = self.predictor_models["amp_scale_predictor"].scale_amp(
                x
            )
            _, z_scaled_sequence, _, _ = self.apply_encoder(x_scaled, dataset, subject)
            return_values["amp_scale_predictor"] = self.predictor_models[
                "amp_scale_predictor"
            ](z_scaled_sequence, scale_label)

        if "vad_classifier" in self.predictor_models and "speech" in inputs:
            vad_labels = inputs["speech"]  # [B, T]

            if vad_labels.shape[-1] != z_sequence.shape[1]:
                # Downsample labels to match number of encoder output embeddings
                vad_labels = F.interpolate(
                    vad_labels.unsqueeze(1),  # [B, 1, T]
                    size=z_sequence.shape[1],  # T2
                ).squeeze(1)  # [B, T2]

            return_values["vad"] = self.predictor_models["vad_classifier"](
                z_independent, vad_labels.flatten(start_dim=0, end_dim=-1)
            )

        if "voiced_classifier" in self.predictor_models and "voicing" in inputs:
            voiced_labels = inputs["voicing"]
            return_values["voiced"] = self.predictor_models["voiced_classifier"](
                z_sequence, voiced_labels
            )

        return return_values

    # NOTE each batch in training_step is a tuple of batches from each of the dataloaders 
    def training_step(self, batch, batch_idx):
        step1_optim, optim, conf_optim, dm_optim = self.optimizers()
        alpha = self.rep_config["alpha"]
        beta = self.rep_config["beta"]

        #TODO remove after debugging
        # with torch.autograd.detect_anomaly():

        ## train main encoder
        if self.current_epoch < self.epoch_stage_1:
            #TODO implement normalizing total batch size across all 3 dataloaders to 32
            # skipping for now to get the framework up and running
            # also using MEGalodon loss instead of regressor loss criterion
            step1_optim.zero_grad()
            task_loss = 0
            domain_loss = 0
            batch_size = 0
            subset = 0
            for idx, batch_i in enumerate(batch):
                if idx == 0:
                    subset = np.random.randint(1, len(batch_i["data"]) - 1)
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = len(batch_i["data"]) - subset
                    batch_i = self._take_subset(batch_i, subset)
                batch_size += subset
                t_loss, losses, metrics, features = self._shared_step(batch_i, batch_idx, "train")
                d_pred = self.domain_classifier(features)
                # d_target = torch.full_like(batch_i['data'], get_dset_encoding(batch_i["info"]["dataset"][0])).to(self.device)
                # d_target = torch.ones((len(batch_i["data"]), 1)) * get_dset_encoding(batch_i["info"]["dataset"][0])
                d_target = torch.zeros((subset, len(batch))).to(self.device)
                d_target[:, idx] = 1
                # d_target = d_target.int()
                # d_target.to(self.device)
                d_loss = self.domain_criterion(d_pred, d_target)
                if t_loss is not None:
                    task_loss += t_loss
                domain_loss += d_loss
            #TODO possibly avg loss over num datasets?
            loss = task_loss + alpha * domain_loss
            self.manual_backward(loss)
            step1_optim.step()

            self.log(
                "train_loss",
                task_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            #TODO maybye only start logging this during second stage
            self.log(
                "domain_train_loss",
                domain_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

            del loss, task_loss, domain_loss
            torch.cuda.empty_cache()

            return 

        ## begin unlearning
        else:
            # update encoder / task heads
            optim.zero_grad()
            task_loss = 0
            batch_vals = []
            batch_size = 0
            subset = 0
            for idx, batch_i in enumerate(batch):
                if idx == 0:
                    subset = np.random.randint(1, len(batch_i["data"]) - 1)
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = len(batch_i["data"]) - subset
                    batch_i = self._take_subset(batch_i, subset)
                batch_size += len(batch_i["data"])
                t_loss, losses, metrics, features = self._shared_step(batch_i, batch_idx, "train")
                # d_target = torch.full_like(batch_i['data'], get_dset_encoding(batch_i["info"]["dataset"][0])).to(self.device)
                # d_target = torch.ones((len(batch_i["data"]), 1)) * get_dset_encoding(batch_i["info"]["dataset"][0])
                d_target = torch.zeros((subset, len(batch))).to(self.device)
                d_target[:, idx] = 1
                # d_target = d_target.int()
                # d_target.to(self.device)
                batch_vals.append({"features": features, "d_target": d_target})
                if t_loss is not None:
                    task_loss += t_loss
            self.manual_backward(task_loss, retain_graph=True)
            optim.step()

            # update just domain classifier
            dm_optim.zero_grad()
            domain_loss = 0
            for vals in batch_vals:
                feats, targets = vals.values()
                d_preds = self.domain_classifier(feats.detach())
                d_loss = self.domain_criterion(d_preds, targets)
                domain_loss += d_loss
            domain_loss = alpha * domain_loss
            self.manual_backward(domain_loss)
            dm_optim.step()

            # update just encoder using domain loss
            conf_optim.zero_grad()
            confusion_loss = 0
            for vals in batch_vals:
                feats, targets = vals.values()
                conf_preds = self.domain_classifier(feats)
                conf_loss = self.conf_criterion(conf_preds, targets)
                confusion_loss += conf_loss
            confusion_loss = beta * confusion_loss
            self.manual_backward(confusion_loss, retain_graph=False) #causing the error - test out in interactive session with unlearning step immediately 
            conf_optim.step()

            self.log(
                "train_loss",
                task_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "domain_train_loss",
                domain_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "confusion_train_loss",
                confusion_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

            del task_loss, domain_loss, confusion_loss
            torch.cuda.empty_cache()

            return 

    def _take_subset(self, batch, subset):
        for key, value in batch.items():
            if key == "info":
                batch[key]= self._take_subset(value, subset)
            else:
                batch[key] = value[:subset]
        return batch

    def validation_step(self, batch, batch_idx):
        #TODO implement normalizing total batch size across all 3 dataloaders to 32
        # skipping for now to get the framework up and running
        # also using MEGalodon loss instead of regressor loss criterion

        if self.current_epoch < self.epoch_stage_1: #TODO make sure diff val functions serve a purpose, check if bypassing hooks avoid the tensor edited in place error
            task_loss = 0
            domain_loss = 0
            batch_size = 0
            domain_preds = []
            domain_targets = []
            subset = 0
            for idx, batch_i in enumerate(batch):
                if idx == 0:
                    subset = np.random.randint(1, len(batch_i["data"]) - 1)
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = len(batch_i["data"]) - subset
                    batch_i = self._take_subset(batch_i, subset)
                batch_size += subset

                t_loss, losses, metrics, features = self._shared_step(batch_i, batch_idx, "val")

                d_pred = self.domain_classifier(features) 

                d_target = torch.zeros((subset, len(batch))).to(self.device)
                d_target[:, idx] = 1

                d_loss = self.domain_criterion(d_pred, d_target)

                domain_preds.append(d_pred)
                domain_targets.append(d_target)
                if t_loss is not None:
                    task_loss += t_loss
                domain_loss += d_loss
            domain_preds = torch.cat(domain_preds, 0)
            domain_targets = torch.cat(domain_targets, 0)

            pred_domains = np.argmax(domain_preds.detach().cpu().numpy(), axis=1)
            true_domains = np.argmax(domain_targets.detach().cpu().numpy(), axis=1)

            acc = accuracy_score(true_domains, pred_domains)
        
            self.log(
                "val_loss",
                task_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "domain_val_loss",
                domain_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "classifier_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

            del domain_preds, domain_targets, pred_domains, true_domains, task_loss, domain_loss 
            torch.cuda.empty_cache()

            return 
        
        else:
            task_loss = 0
            batch_size = 0
            domain_preds = []
            domain_targets = []
            subset = 0
            for idx, batch_i in enumerate(batch):
                if idx == 0:
                    subset = np.random.randint(1, len(batch_i["data"]) - 1)
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = len(batch_i["data"]) - subset
                    batch_i = self._take_subset(batch_i, subset)
                batch_size += subset

                t_loss, losses, metrics, features = self._shared_step(batch_i, batch_idx, "val")

                # explicitly call forward to avoid hooks
                d_pred = self.domain_classifier.forward(features) 

                d_target = torch.zeros((subset, len(batch))).to(self.device)
                d_target[:, idx] = 1

                domain_preds.append(d_pred)
                domain_targets.append(d_target)
                if t_loss is not None:
                    task_loss += t_loss
            domain_preds = torch.cat(domain_preds, 0)
            domain_targets = torch.cat(domain_targets, 0)

            pred_domains = np.argmax(domain_preds.detach().cpu().numpy(), axis=1)
            true_domains = np.argmax(domain_targets.detach().cpu().numpy(), axis=1)

            acc = accuracy_score(true_domains, pred_domains)
        
            self.log(
                "val_loss",
                task_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "classifier_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

            del domain_preds, domain_targets, pred_domains, true_domains, task_loss 
            torch.cuda.empty_cache()

            return


    #TODO implement domain unlearning iterative training scheme
    def test_step(self, batch, batch_idx):
        #TODO implement normalizing total batch size across all 3 dataloaders to 32
        # skipping for now to get the framework up and running
        # also using MEGalodon loss instead of regressor loss criterion
        task_loss = 0
        batch_size = 0
        domain_preds = []
        domain_targets = []
        subset = 0
        for idx, batch_i in enumerate(batch):
            if idx == 0:
                subset = np.random.randint(1, len(batch_i["data"]) - 1)
                batch_i = self._take_subset(batch_i, subset)
            elif idx == 1:
                subset = len(batch_i["data"]) - subset
                batch_i = self._take_subset(batch_i, subset)
            batch_size += subset
            t_loss, losses, metrics, features = self._shared_step(batch_i, batch_idx, "test")
            d_pred = self.domain_classifier(features)
            # d_target = torch.full_like(batch_i["data"], get_dset_encoding(batch_i["info"]["dataset"][0])).to(self.device)
            # d_target = torch.ones((len(batch_i["data"]), 1)) * get_dset_encoding(batch_i["info"]["dataset"][0])
            d_target = torch.zeros((subset, len(batch))).to(self.device)
            d_target[:, idx] = 1
            # d_target = d_target.int()
            # d_target.to(self.device)
            domain_preds.append(d_pred)
            domain_targets.append(d_target)
            if t_loss is not None:
                task_loss += t_loss
        domain_preds = torch.cat(domain_preds, 0)
        domain_targets = torch.cat(domain_targets, 0)

        pred_domains = np.argmax(domain_preds.detach().cpu().numpy(), axis=1)
        true_domains = np.argmax(domain_targets.detach().cpu().numpy(), axis=1)
        acc = accuracy_score(true_domains, pred_domains)
    
        self.log(
            "test_loss",
            task_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "classifier_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        del domain_preds, domain_targets, pred_domains, true_domains, task_loss 
        torch.cuda.empty_cache()

        return 

    def _shared_step(self, batch, batch_idx, stage: str):
        loss = 0.0
        losses = {}
        metrics = {}

        data_key = get_key_from_batch_identifier(batch["info"])
        dataset = batch["info"]["dataset"][0]

        return_values = self(batch)
        features = return_values.pop("classifier features")

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

        if loss == 0.0:
            loss = None

        return loss, losses, metrics, features

    def configure_optimizers(self):
        encoder_params = list(filter(lambda p: p.requires_grad, self.encoder_models.parameters()))
        predictor_params = list(filter(lambda p: p.requires_grad, self.predictor_models.parameters()))
        domain_classifier_params = list(filter(lambda p: p.requires_grad, self.domain_classifier.parameters()))

        step1_optim = torch.optim.AdamW(
            encoder_params + predictor_params + domain_classifier_params, 
            lr=self.learning_rate
        )
        optim = torch.optim.Adam(encoder_params + predictor_params, lr=1e-4)
        conf_optim = torch.optim.Adam(encoder_params, lr=1e-4)
        dm_optim = torch.optim.Adam(domain_classifier_params, lr=1e-4)
        
        return step1_optim, optim, conf_optim, dm_optim
    
    def finetuning_mode(self):
        self.freeze_except(
            ["dataset_block", "subject_"]
        )  # Keep these weights open to fine-tune with new datasets
        self.disable_ssl()
        self.disable_classifiers()

    def freeze_except(self, module_names):
        if isinstance(module_names, str):
            module_names = [module_names]

        for key in self.encoder_models.keys():
            for module_name in module_names:
                if module_name not in key:
                    for param in self.encoder_models[key].parameters():
                        param.requires_grad = False

        for key in self.predictor_models.keys():
            for module_name in module_names:
                if module_name not in key:
                    for param in self.predictor_models[key].parameters():
                        param.requires_grad = False
        
        if "domain_classifier" not in module_names:
            for param in self.domain_classifier.parameters():
                param.requires_grad = False

    def disable_ssl(self):
        keys = list(self.encoder_models.keys())
        for key in keys:
            if "predictor" in key:
                self.encoder_models.pop(key)

        keys = list(self.predictor_models.keys())
        for key in keys:
            if "predictor" in key:
                self.predictor_models.pop(key)

        # Also remove the SSL projector for fine-tuning
        self.encoder_models["projector"] = nn.Identity()

    def disable_classifiers(self):
        keys = list(self.encoder_models.keys())
        for key in keys:
            if "classifier" in key:
                self.encoder_models.pop(key)

        keys = list(self.predictor_models.keys())
        for key in keys:
            if "classifier" in key:
                self.predictor_models.pop(key)

    def add_classifier(self, classifier_type: str, params: dict):
        # Labeled tasks for representation shaping or downstream classification

        if classifier_type == "vad_classifier":
            self.weightings["vad"] = params.get("weight", 1.0)
            params.pop("weight", None)
            if "subject_embedding" in self.rep_config:
                params["input_dim"] += self.rep_config["subject_embedding"][
                    "embedding_dim"
                ]
            self.predictor_models.update({"vad_classifier": VADClassifier(**params)})
        elif classifier_type == "vad_classifier_linear":
            self.weightings["vad"] = params.get("weight", 1.0)
            params.pop("weight", None)
            if "subject_embedding" in self.rep_config:
                params["input_dim"] += self.rep_config["subject_embedding"][
                    "embedding_dim"
                ]
            self.predictor_models.update({"vad_classifier": VADClassifierLinear(**params)})

        if classifier_type == "voiced_classifier":
            self.weightings["voiced"] = params.get("weight", 1.0)
            params.pop("weight", None)

            if "subject_embedding" in self.rep_config:
                params["input_dim"] += self.rep_config["subject_embedding"][
                    "embedding_dim"
                ]

            if params["type"] == "mlp":
                del params["type"]
                self.predictor_models.update(
                    {"voiced_classifier": VoicedClassifierMLP(**params)}
                )
            elif params["type"] == "lstm":
                del params["type"]
                self.predictor_models.update(
                    {"voiced_classifier": VoicedClassifierLSTM(**params)}
                )
            elif params["type"] == "linear":
                del params["type"]
                self.predictor_models.update(
                    {"voiced_classifier": VoicedClassifierLinear(**params)}
                )
            else:
                raise ValueError("Voiced classifier type not recognised")


if __name__ == "__main__":
    model = RepLearnerUnlearner(
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
