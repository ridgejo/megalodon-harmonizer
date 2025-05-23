import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from dataloaders.data_utils import get_key_from_batch_identifier, get_dset_encoding, get_age_distribution_labels
from dataloaders.constants import MOUS_AGES, CAMCAN_AGES
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
from models.domain_classifier import DomainClassifier, LSVM_DomainClassifier
from models.confusion_loss import ConfusionLoss
from models.analysis_utils import plot_tsne
from models.sam_optimizer import SAM

class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args):
        return self.func(*args)


class RepHarmonizer(L.LightningModule):
    """
    Representation learner with adversarial harmonization, modified to support 
    simultaneous harmonization of different confounds (e.g. participant age and dataset).
    """

    def __init__(self, rep_config):
        super().__init__()
        self.automatic_optimization = False
        ## Load in special args from command line ##
        self.epoch_stage_1 = rep_config["epoch_stage_1"]
        self.max_epochs = rep_config["max_epochs"]
        self.batch_size = rep_config["batch_size"]
        self.num_feats = rep_config.get("num_classifier_feats", rep_config["dataset_block"]["shared_dim"])
        self.run_name = rep_config.get("run_name", "")
        self.multi_dm_pred = rep_config.get("multi_dm_pred", False)
        self.agg_task_feats = rep_config.get("agg_task_feats", False)
        self.age_confound = rep_config.get("age_confound", False)
        self.learning_rate = rep_config["lr"]
        self.dm_learning_rate = rep_config.get("dm_lr", 0.0001)
        self.conf_learning_rate = rep_config.get("conf_lr", 0.0001)
        self.task_learning_rate = rep_config.get("task_lr", 0.0001)
        self.tsne = rep_config.get("tsne", False)
        self.sdat = rep_config.get("sdat", False) ## buggy
        self.sgd = rep_config.get("sgd", False)
        self.full_run = rep_config.get("full_run", False)
        self.clear_optim = rep_config.get("clear_optim", False)
        self.clear_betas = rep_config.get("clear_betas", False)
        self.finetune = rep_config.get("finetune", False)
        self.no_dm_control = rep_config.get("no_dm_control", False)
        self.intersect_only = rep_config.get("intersect_only", False) ## not currently supported with multi confound harmonization
        self.no_proj_encode = rep_config.get("no_proj_encode", True)
        
        # Hacky fix to support loading checkpoints trained with different batch sizes
        batch_dim = rep_config.get("batch_dim")
        if batch_dim is None:
            batch_dim = self.batch_size
        
        self.activations = None
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
            if self.no_proj_encode:
                predictor_models["projector"] = Projector(**rep_config["projector"])
            else:
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
        domain_classifiers = {}
        if batch_dim == -1:
            self.domain_classifiers = None
        elif rep_config.get("lsvm") is not None:
            domain_classifiers["lsvm"] =  LSVM_DomainClassifier(self.batch_size, rep_config.get("num_datasets", 2)) # was 2560
        elif self.multi_dm_pred:
            # all for dataset bias
            domain_classifiers["backbone"] = DomainClassifier(nodes=rep_config.get("num_datasets", 2), init_features=self.num_feats, batch_size=batch_dim)
            domain_classifiers["band_predictor"] = DomainClassifier(nodes=rep_config.get("num_datasets", 2), init_features=self.num_feats, batch_size=batch_dim)
            domain_classifiers["phase_diff"] = DomainClassifier(nodes=rep_config.get("num_datasets", 2), init_features=self.num_feats, batch_size=batch_dim)
            domain_classifiers["amp_scale"] = DomainClassifier(nodes=rep_config.get("num_datasets", 2), init_features=self.num_feats, batch_size=batch_dim)
            domain_classifiers = nn.ModuleDict(domain_classifiers)
            self.domain_classifiers = nn.ModuleDict({"dataset":domain_classifiers})
        elif not rep_config.get("no_dataset", False):
            domain_classifiers["dataset"] = DomainClassifier(nodes=rep_config.get("num_datasets", 2), init_features=self.num_feats, batch_size=batch_dim) 
        if self.age_confound:
            # nodes (output dim) specific to MOUS + Cam-CAN)
            # 72 possible ages from youngest (18) to oldest (89 - from Cam-CAN datatset)
            domain_classifiers["age"] = DomainClassifier(nodes=72, init_features=self.num_feats, batch_size=batch_dim) 
        self.domain_classifiers = nn.ModuleDict(domain_classifiers)
        domain_criterions = {}
        if self.age_confound:
            domain_criterions["age"] = nn.KLDivLoss(reduction='batchmean')
        domain_criterions["dataset"] = nn.CrossEntropyLoss() 
        self.domain_criterions = nn.ModuleDict(domain_criterions)
        self.conf_criterion = ConfusionLoss()
        self.rep_config = rep_config

        # Add classifiers if used in pre-training
        for k, v in rep_config.items():
            if "classifier" in k:
                self.add_classifier(k, v)

    ### NOTE ###
    # 'task' is not the most intuitive value for stage anymore
    # refers to cloning all params and implementing functional variants on
    # forward passes to avoid errors with backprop and multiple optimizers 
    # in a single training step. 
    # Set to 'encode' if you want to avoid this behavior, though without
    # further changes training will fail during adversarial harmonization
    def apply_encoder(self, z, dataset, subject, stage="task"):        
        z = self.encoder_models["dataset_block"](z, dataset, stage=stage)
        z = self.encoder_models["encoder"](z, stage=stage)
        z = self.encoder_models["transformer"](z)
        z, _, commit_loss = self.encoder_models["quantize"](z)

        # Generic subject embedding
        subject_embedding = self.encoder_models["subject_embedding"](dataset, subject)

        # Subject block
        z = self.encoder_models["subject_block"](z, dataset, subject)

        # Subject FiLM conditioning
        z = self.encoder_models["subject_film_module"](z, subject_embedding, stage=stage)

        # Subject embedding concatentation
        z = self.encoder_models["attach_subject"](z, subject_embedding)

        # Max Pooling over the entire time dimension T
        maxpool = nn.MaxPool1d(kernel_size=z.shape[2])  # Pool across the time dimension
        pooled_data = maxpool(z)  # Resulting shape will be [B, E, 1]

        # Squeeze the last dimension to get [B, E]
        features = pooled_data.squeeze(-1)  # Shape [B, E]

        # Create two different views for sequence models and independent classifiers
        z_sequence = z.permute(0, 2, 1)  # [B, T, E]
        z_independent = z_sequence.flatten(start_dim=0, end_dim=1)  # [B * T, E]

        # Apply SSL projector to z_sequence
        T, E = z_sequence.shape[1:]

        if self.no_proj_encode:
            z_sequence = torch.unflatten(
                self.predictor_models["projector"](
                    z_sequence.flatten(start_dim=1, end_dim=-1)
                ),
                dim=-1,
                sizes=(T, E),
            )
        elif self.finetune:
            z_sequence = torch.unflatten(
                self.encoder_models["projector"](
                    z_sequence.flatten(start_dim=1, end_dim=-1)
                ),
                dim=-1,
                sizes=(T, E),
            )
        else:
            z_sequence = torch.unflatten(
                self.encoder_models["projector"](
                    z_sequence.flatten(start_dim=1, end_dim=-1), stage=stage
                ),
                dim=-1,
                sizes=(T, E),
            )

        return features, z_sequence, z_independent, commit_loss

    def forward(self, inputs):
        x = inputs["data"]
        sensor_pos = None

        dataset = inputs["info"]["dataset"][0]
        subject = inputs["info"]["subject_id"]

        return_values = {}
        features = {}
        backbone_feats, z_sequence, z_independent, commit_loss = self.apply_encoder(x, dataset, subject)
        return_values["quantization"] = {"commit_loss": commit_loss}
        features["backbone"] = backbone_feats

        if "band_predictor" in self.predictor_models:
            x_filtered, band_label = self.predictor_models["band_predictor"].filter_band(
                x, sample_rate=250
            )  # warning: hardcoded
            filtered_feats, z_filtered_sequence, _, _ = self.apply_encoder(x_filtered, dataset, subject)
            return_values["band_predictor"] = self.predictor_models["band_predictor"](
                z_filtered_sequence, band_label
            )
            features["band_predictor"] = filtered_feats

        if "phase_diff_predictor" in self.predictor_models:
            x_shifted, phase_label = self.predictor_models[
                "phase_diff_predictor"
            ].apply_random_phase_shift(x)
            shifted_feats, z_shifted_sequence, _, _ = self.apply_encoder(x_shifted, dataset, subject)
            return_values["phase_diff_predictor"] = self.predictor_models[
                "phase_diff_predictor"
            ](z_shifted_sequence, phase_label)
            features["phase_diff"] = shifted_feats

        if "masked_channel_predictor" in self.predictor_models:
            x_masked, mask_label = self.predictor_models[
                "masked_channel_predictor"
            ].mask_input(x, sensor_pos)
            masked_feats, z_mask_sequence, _, _ = self.apply_encoder(x_masked, dataset, subject)
            return_values["masked_channel_pred"] = self.predictor_models[
                "masked_channel_predictor"
            ](z_mask_sequence, mask_label)
            features["masked_channel"] = masked_feats

        if "amp_scale_predictor" in self.predictor_models:
            x_scaled, scale_label = self.predictor_models["amp_scale_predictor"].scale_amp(
                x
            )
            scaled_feats, z_scaled_sequence, _, _ = self.apply_encoder(x_scaled, dataset, subject)
            return_values["amp_scale_predictor"] = self.predictor_models[
                "amp_scale_predictor"
            ](z_scaled_sequence, scale_label)
            features["amp_scale"] = scaled_feats

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

        if self.agg_task_feats:
            return features, return_values
        else:
            return features["backbone"], return_values
    
    def _encode(self, batch):
        x = batch["data"]
        dataset = batch["info"]["dataset"][0]
        subject = batch["info"]["subject_id"]

        features, _, _, _ = self.apply_encoder(x, dataset, subject)

        return features

    def _shared_step(self, batch, stage="train"):
        loss = 0.0
        losses = {}
        metrics = {}

        data_key = get_key_from_batch_identifier(batch["info"])
        dataset = batch["info"]["dataset"][0]

        features, return_values = self(batch)

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

        return features, loss, losses, metrics

    ## helper function for age confound harmonization
    ## NOTE specific to MOUS and schoffelen datasets
    def get_age_targets(self, subjects, dataset):
        if dataset == "shafto2014":
            age_dict = CAMCAN_AGES
        if dataset == "schoffelen2019":
            age_dict = MOUS_AGES

        subject_ids = list(age_dict.keys())
        ages = list(age_dict.values())

        # map subjects to their index
        id_to_index = {id: idx for idx, id in enumerate(subject_ids)}

        # convert ages to a tensor
        age_tensor = torch.tensor(ages).to(self.device)

        # get indices of input subject IDs
        subject_indices = torch.tensor([id_to_index[id] for id in subjects]).to(self.device)

        # gather ages from age tensor
        age_targets = age_tensor[subject_indices]

        return age_targets

    # NOTE each batch in training_step is a tuple of batches from each of the dataloaders 
    def training_step(self, batch, batch_idx):
        ## Fine-tuning Mode ##
        if self.finetune:
            ft_optim = self.optimizers()
            ft_optim.zero_grad()
            features, loss, losses, metrics = self._shared_step(batch=batch, stage="train")
            self.manual_backward(loss)
            ft_optim.step()

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
                    sync_dist=True,
                )
                self.log_dict(
                    losses,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )
                self.log_dict(
                    metrics,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

            return loss

        ## Pre-training Mode ##
        if self.sdat:
            step1_optim, optim, dm_optim = self.optimizers()
        else:
            step1_optim, optim, conf_optim, dm_optim = self.optimizers()
        alpha = self.rep_config["alpha"]
        beta = self.rep_config["beta"]

        # If datasets being harmonized are heavily biased by demographics, harmonize only over intersections
        # Assumes intersect loader is specified last in config file
        if self.intersect_only:
            intersect_batch = batch[-1]
            batch = batch[:-1]
        
        ## Train main encoder ##
        # Warm-up Phase #
        if self.current_epoch < self.epoch_stage_1: 
            step1_optim.zero_grad()
            task_loss = 0
            domain_loss = 0
            batch_size = 0
            subset = 0
            split_1 = 0
            domain_preds = {}
            domain_targets = {}
            for idx, batch_i in enumerate(batch):
                if len(batch_i["data"]) < self.batch_size:
                    print(f"Train dataset {idx} batch is less than batch_size", flush=True)
                if len(batch) == 2:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        while subset >= len(batch[1]["data"]): ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = len(batch_i["data"]) - subset
                        batch_i = self._take_subset(batch_i, subset)
                elif len(batch) == 3:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        while subset >= len(batch[1]["data"]) - 2: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        while subset >= len(batch[2]["data"]) - split_1: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 2:
                        subset = len(batch_i["data"]) - split_1 - subset
                        batch_i = self._take_subset(batch_i, subset)
                
                batch_size += subset
                features, t_loss, losses, metrics = self._shared_step(batch=batch_i, stage="train")
                if self.intersect_only and idx == 0:
                    intersect_batch = self._take_subset(intersect_batch, split_1)
                    features = self._encode(intersect_batch)
                
                # collect preds and targets for all counfounds
                for confound in self.domain_classifiers.keys(): 
                    if domain_preds.get(confound) is None:
                        if self.agg_task_feats:
                            domain_preds[confound] = {}
                        else:
                            domain_preds[confound] = []

                    if self.agg_task_feats: 
                        for key, feats in features.items():
                            if self.multi_dm_pred:
                                if domain_preds[confound].get(key) is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound][key](feats)] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound][key](feats))
                            else:
                                if domain_preds[confound].get(key) is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound](feats)] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound](feats))
                    else:
                        d_pred = self.domain_classifiers[confound](features)
                        domain_preds[confound].append(d_pred)

                    if domain_targets.get(confound) is None:
                        domain_targets[confound] = []

                    if confound == "age":
                        d_target = self.get_age_targets(batch_i["info"]["subject"], batch_i["info"]["dataset"][0])
                    else:
                        d_target = torch.full((subset,), idx).to(self.device)
                    domain_targets[confound].append(d_target)

                if t_loss is not None:
                    task_loss += t_loss

            domain_loss = 0
            for confound in self.domain_classifiers.keys():    
                domain_targets[confound] = torch.cat(domain_targets[confound])
                if confound == "age":
                    domain_targets[confound] = get_age_distribution_labels(domain_targets[confound]).to(self.device)
                    domain_targets[confound] = F.softmax(domain_targets[confound], dim=1)
                if self.agg_task_feats:
                    for key, pred_list in domain_preds[confound].items():
                        preds = torch.cat(pred_list)
                        if confound == "age":
                            preds = torch.softmax(preds, dim=1)
                            preds = torch.argmax(preds, dim=1) + 18
                            preds = get_age_distribution_labels(preds).to(self.device)
                            preds = F.log_softmax(preds, dim=1)
                        domain_loss += self.domain_criterions[confound](preds, domain_targets[confound])
                else:   
                    domain_preds[confound] = torch.cat(domain_preds[confound])
                    if confound == "age":
                        domain_preds[confound] = torch.softmax(domain_preds[confound], dim=1)
                        domain_preds[confound] = torch.argmax(domain_preds[confound], dim=1) + 18
                        domain_preds[confound] = get_age_distribution_labels(domain_preds[confound]).to(self.device)
                        domain_preds[confound] = F.log_softmax(domain_preds[confound], dim=1)
                    domain_loss += self.domain_criterions[confound](domain_preds[confound], domain_targets[confound])

            if self.no_dm_control:
                loss = task_loss
            else:
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

            # clear mem just in case
            del loss, task_loss, domain_loss, domain_preds, domain_targets
            torch.cuda.empty_cache()

            return 

        ## Harmonization Phase ##
        else:
            # update encoder + task heads #
            optim.zero_grad()
            task_loss = 0
            batch_vals = {}
            batch_size = 0
            subset = 0
            split_1 = 0
            for idx, batch_i in enumerate(batch):
                if len(batch_i["data"]) < self.batch_size:
                    print(f"Train dataset {idx} batch is less than batch_size", flush=True)
                if len(batch) == 2:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        if self.intersect_only:
                            while subset >= len(batch[1]["data"]) - 1 or subset > len(intersect_batch["data"]): ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        else:
                            while subset >= len(batch[1]["data"]) - 1: ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = len(batch_i["data"]) - subset
                        batch_i = self._take_subset(batch_i, subset)
                elif len(batch) == 3:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        if self.intersect_only:
                            while subset >= len(batch[1]["data"]) - 2 or subset > len(intersect_batch["data"]): ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        else:
                            while subset >= len(batch[1]["data"]) - 2: ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        while subset >= len(batch[2]["data"]) - split_1: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 2:
                        subset = len(batch_i["data"]) - split_1 - subset
                        batch_i = self._take_subset(batch_i, subset)
                batch_size += len(batch_i["data"])

                features, t_loss, losses, metrics = self._shared_step(batch=batch_i, stage="train")

                for confound in self.domain_classifiers.keys():
                    if batch_vals.get(confound) is None:
                        batch_vals[confound] = []

                    if confound == "age":
                        d_target = self.get_age_targets(batch_i["info"]["subject"], batch_i["info"]["dataset"][0])
                    else:
                        d_target = torch.full((subset,), idx).to(self.device)
                    batch_vals[confound].append((features, d_target))
                if t_loss is not None:
                    task_loss += t_loss

            self.manual_backward(task_loss, retain_graph=True)

            if self.sdat:
                optim.first_step(zero_grad=True)
            else:
                optim.step()   

            # update just domain classifier #
            dm_optim.zero_grad()
            domain_loss = 0
            domain_preds = {}
            domain_targets = {}

            if self.intersect_only:
                if len(intersect_batch["data"]) < self.batch_size:
                    print(f"Intersect batch is less than batch_size", flush=True)
                # relies heavily on assumption that Shafto is first
                intersect_batch = self._take_subset(intersect_batch, split_1)
                features = self._encode(intersect_batch)
                targets = torch.full((split_1,), 0).to(self.device)
                batch_vals[0] = (features, targets)

            for confound in self.domain_classifiers.keys():
                if domain_preds.get(confound) is None:
                    if self.agg_task_feats:
                        domain_preds[confound] = {}
                    else:
                        domain_preds[confound] = []
                if domain_targets.get(confound) is None:
                    domain_targets[confound] = []

                for features, targets in batch_vals[confound]:
                    if self.agg_task_feats:
                        for key, feats in features.items():
                            if self.multi_dm_pred:
                                if domain_preds[confound].get(key) is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound][key](feats.detach())] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound][key](feats.detach()))
                            else:
                                if domain_preds[confound].get(key) is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound](feats.detach())] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound](feats.detach()))
                    else:
                        domain_preds[confound].append(self.domain_classifiers[confound](features.detach()))
                    domain_targets[confound].append(targets)

            domain_loss = 0
            for confound in self.domain_classifiers.keys(): 
                domain_targets[confound] = torch.cat(domain_targets[confound])
                if confound == "age":
                    domain_targets[confound] = get_age_distribution_labels(domain_targets[confound]).to(self.device)
                    domain_targets[confound] = F.softmax(domain_targets[confound], dim=1)
                if self.agg_task_feats:
                    for key, pred_list in domain_preds[confound].items():
                        preds = torch.cat(pred_list)
                        if confound == "age":
                            preds = torch.softmax(preds, dim=1)
                            preds = torch.argmax(preds, dim=1) + 18
                            preds = get_age_distribution_labels(preds).to(self.device)
                            preds = F.log_softmax(preds, dim=1)
                        domain_loss += self.domain_criterions[confound](preds, domain_targets[confound])
                else:
                    domain_preds[confound] = torch.cat(domain_preds[confound])
                    if confound == "age":
                        domain_preds[confound] = torch.softmax(domain_preds[confound], dim=1)
                        domain_preds[confound] = torch.argmax(domain_preds[confound], dim=1) + 18
                        domain_preds[confound] = get_age_distribution_labels(domain_preds[confound]).to(self.device)
                        domain_preds[confound] = F.log_softmax(domain_preds[confound], dim=1)
                    domain_loss += self.domain_criterions[confound](domain_preds[confound], domain_targets[confound])

            domain_loss = alpha * domain_loss
            self.manual_backward(domain_loss)

            dm_optim.step()

            # update just encoder using domain loss #
            if not self.sdat:
                conf_optim.zero_grad()
            confusion_loss = 0


            domain_preds = {}
            domain_preds = {}

            for confound in self.domain_classifiers.keys():
                if domain_preds.get(confound) is None:
                    if self.agg_task_feats:
                        domain_preds[confound] = {}
                    else:
                        domain_preds[confound] = []

                for features, targets in batch_vals[confound]:
                    if self.agg_task_feats:
                        for key, feats in features.items():
                            pred_list = domain_preds[confound].get(key)
                            if self.multi_dm_pred:
                                if pred_list is None:
                                    pred = self.domain_classifiers[confound][key](feats)
                                    pred = torch.softmax(pred, dim=1)
                                    domain_preds[confound][key] = [pred] 
                                else:
                                    pred = self.domain_classifiers[confound][key](feats)
                                    pred = torch.softmax(pred, dim=1)
                                    domain_preds[confound][key].append(pred)
                            else:
                                if pred_list is None:
                                    pred = self.domain_classifiers[confound](feats)
                                    pred = torch.softmax(pred, dim=1)
                                    domain_preds[confound][key] = [pred] 
                                else:
                                    pred = self.domain_classifiers[confound](feats)
                                    pred = torch.softmax(pred, dim=1)
                                    domain_preds[confound][key].append(pred)
                    else:
                        conf_preds = self.domain_classifiers[confound](features)
                        conf_preds = torch.softmax(conf_preds, dim=1)
                        domain_preds.append(conf_preds)
            
            confusion_loss = 0
            for confound in self.domain_classifiers.keys():
                if self.agg_task_feats:
                    for key, pred_list in domain_preds[confound].items():
                        preds = torch.cat(pred_list)
                        confusion_loss += self.conf_criterion(preds, domain_targets[confound])
                else:
                    domain_preds[confound] = torch.cat(domain_preds[confound])
                    confusion_loss += self.conf_criterion(domain_preds[confound], domain_targets[confound])

            confusion_loss = beta * confusion_loss

            # Check for NaNs in confusion_loss before backward call in except loop
            if torch.isnan(confusion_loss).any():
                raise ValueError("NaN detected in confusion_loss before backward call ")
            if torch.isinf(confusion_loss).any():
                raise ValueError("Inf detected in confusion_loss before backward call ")
            
            self.manual_backward(confusion_loss, retain_graph=False) 
            
            if self.sdat:
                optim.second_step(zero_grad=True)
            else:
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

    ## Helper function taking batch slices ##
    def _take_subset(self, batch, subset):
        for key, value in batch.items():
            if key == "info":
                batch[key]= self._take_subset(value, subset)
            else:
                batch[key] = value[:subset]
        return batch
    
    def _pad_subset(self, batch, subset):
        for key, value in batch.items():
            if key == "info":
                batch[key]= self._take_subset(value, subset)
            else:
                pad = value[:subset]
                batch[key] = torch.cat((value, pad))
        return batch

    def validation_step(self, batch, batch_idx):
        # Fine-tuning Mode #
        if self.finetune:
            features, loss, losses, metrics = self._shared_step(batch=batch, stage="val")
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
                    sync_dist=True,
                )
                self.log_dict(
                    losses,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )
                self.log_dict(
                    metrics,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

            return loss
        
        # If datasets being harmonized are heavily biased by demographics, harmonize only over intersections
        # Assumes intersect loader is specified last in config file
        if self.intersect_only:
            intersect_batch = batch[-1]
            batch = batch[:-1]

        ## Warm-up Phase ##
        if self.current_epoch < self.epoch_stage_1: 
            if self.full_run and self.current_epoch == self.epoch_stage_1 - 1 and batch_idx == 0:
                save_activations = True
            else:
                save_activations = False
            task_loss = 0
            domain_loss = 0
            batch_size = 0
            domain_preds = {}
            domain_targets = {}
            if self.age_confound: 
                acc_targets = []
            subset = 0
            split_1 = 0
            if save_activations:
                activations = []
            for idx, batch_i in enumerate(batch):
                if len(batch_i["data"]) < self.batch_size:
                    print(f"Val dataset {idx} batch is less than batch_size", flush=True)
                if len(batch) == 2:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        # if self.intersect_only:
                        #     while subset >= len(batch[1]["data"]) or subset > len(intersect_batch["data"]): ## hacky fix for abnormal batch sizes
                        #         subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        # else:
                        while subset >= len(batch[1]["data"]) - 1: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = len(batch_i["data"]) - subset
                        batch_i = self._take_subset(batch_i, subset)
                elif len(batch) == 3:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        # if self.intersect_only:
                        #     while subset >= len(batch[1]["data"]) - 2 or subset > len(intersect_batch["data"]): ## hacky fix for abnormal batch sizes
                        #         subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        # else:
                        while subset >= len(batch[1]["data"]) - 2: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        while subset >= len(batch[2]["data"]) - split_1: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 2:
                        subset = len(batch_i["data"]) - split_1 - subset
                        batch_i = self._take_subset(batch_i, subset)
                batch_size += subset

                features, t_loss, losses, metrics = self._shared_step(batch=batch_i, stage="val")
                
                if self.intersect_only and idx == 0:
                    intersect_batch = self._take_subset(intersect_batch, split_1)
                    features = self._encode(intersect_batch)
                    
                if save_activations:
                    activations.append(features.detach())

                for confound in self.domain_classifiers.keys():
                    if domain_preds.get(confound) is None:
                        if self.agg_task_feats:
                            domain_preds[confound] = {}
                        else:
                            domain_preds[confound] = []
                    if domain_targets.get(confound) is None:
                        domain_targets[confound] = []
                    
                    if self.agg_task_feats:
                        for key, feats in features.items():
                            pred_list = domain_preds[confound].get(key)
                            if self.multi_dm_pred:
                                if pred_list is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound][key].forward(feats)] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound][key].forward(feats))
                            else:
                                if pred_list is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound].forward(feats)] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound].forward(feats))
                    else:
                        d_pred = self.domain_classifiers[confound].forward(features) 
                        domain_preds[confound].append(d_pred)

                    if confound == "age":
                        d_target = self.get_age_targets(batch_i["info"]["subject"], batch_i["info"]["dataset"][0])
                        acc_targets.append(d_target.int())
                    else:
                        d_target = torch.full((subset,), idx).to(self.device)
                    domain_targets[confound].append(d_target)

                if t_loss is not None:
                    task_loss += t_loss

            domain_loss = 0
            acc = 0
            for confound in self.domain_classifiers.keys():
                domain_targets[confound] = torch.cat(domain_targets[confound])
                if confound == "age":
                    domain_targets[confound] = get_age_distribution_labels(domain_targets[confound]).to(self.device)
                    domain_targets[confound] = F.softmax(domain_targets[confound], dim=1)
                if self.agg_task_feats:
                    for key, pred_list in domain_preds[confound].items():
                        preds = torch.cat(pred_list)
                        if confound == "age":
                            loss_preds = torch.softmax(preds, dim=1)
                            loss_preds = torch.argmax(loss_preds, dim=1) + 18 # convert to actual age values instead of indices
                            loss_preds = get_age_distribution_labels(loss_preds).to(self.device)
                            loss_preds = F.log_softmax(loss_preds, dim=1)
                            domain_loss += self.domain_criterions[confound](loss_preds, domain_targets[confound])
                        else:
                            domain_loss += self.domain_criterions[confound](preds, domain_targets[confound])
                        domain_preds[confound][key] = torch.softmax(preds, dim=1)
                else:
                    domain_preds[confound] = torch.cat(domain_preds[confound])
                    if confound == "age":
                        loss_preds = torch.softmax(domain_preds[confound], dim=1)
                        loss_preds = torch.argmax(loss_preds, dim=1) + 18
                        loss_preds = get_age_distribution_labels(loss_preds).to(self.device)
                        loss_preds = F.log_softmax(loss_preds, dim=1)
                        domain_loss += self.domain_criterions[confound](loss_preds, domain_targets[confound])
                    else:
                        domain_loss += self.domain_criterions[confound](domain_preds[confound], domain_targets[confound])
                    domain_preds[confound] = torch.softmax(domain_preds[confound], dim=1)

                if confound == "age":
                    domain_targets[confound] = torch.cat(acc_targets)
                true_domains = domain_targets[confound].detach().cpu().numpy()
                if self.agg_task_feats:
                    accs = []
                    for key, pred_list in domain_preds[confound].items():
                        pred_domains = np.argmax(pred_list.detach().cpu().numpy(), axis=1)
                        if confound == "age":
                            pred_domains = pred_domains + 18
                        accs.append(accuracy_score(true_domains, pred_domains))
                    
                    acc += sum(accs) / len(accs)
                else:
                    pred_domains = np.argmax(domain_preds[confound].detach().cpu().numpy(), axis=1)
                    if confound == "age":
                        pred_domains = pred_domains + 18
                    acc += accuracy_score(true_domains, pred_domains)

            #TODO fix - if not multi then acc needs to be moved below, if is multi then needs overhaul
            if save_activations:
                activations = torch.cat(activations).to("cpu")
                label_mapping = {0: 'dataset_1', 1: 'dataset_2'}
                # Convert numerical labels to class names
                label_names = [label_mapping[label.item()] for label in true_domains]
                save_path = Path("/data/engs-pnpl/wolf6942/experiments/MEGalodon/full_run/fullrun_tsne_plots")
                np.save(save_path / f"{self.run_name}_task_activations.npy", activations.numpy())
                np.save(save_path / f"{self.run_name}_task_labels.npy", np.array(label_names))
                print("Saving activations...")
                plot_tsne(activations=activations, labels=label_names, save_dir=save_path, 
                          file_name=f"{self.run_name}_task_tsne.png")

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
        
        ## Harmonization Phase ##
        else:
            if self.full_run and self.current_epoch == self.max_epochs - 1 and batch_idx == 0:
                save_activations = True
            else:
                save_activations = False
            task_loss = 0
            batch_size = 0
            domain_preds = {}
            domain_targets = {}
            if self.age_confound: 
                acc_targets = []
            subset = 0
            split_1 = 0
            if save_activations:
                activations = []
            for idx, batch_i in enumerate(batch):
                if len(batch_i["data"]) < self.batch_size:
                    print(f"Val dataset {idx} batch is less than batch_size", flush=True)
                if len(batch) == 2:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        if self.intersect_only:
                            while subset >= len(batch[1]["data"]) -1 or subset > len(intersect_batch["data"]): ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        else:
                            while subset >= len(batch[1]["data"]) - 1: ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 1)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = len(batch_i["data"]) - subset
                        batch_i = self._take_subset(batch_i, subset)
                elif len(batch) == 3:
                    if idx == 0:
                        subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        if self.intersect_only:
                            while subset >= len(batch[1]["data"]) - 2 or subset > len(intersect_batch["data"]): ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        else:
                            while subset >= len(batch[1]["data"]) - 2: ## hacky fix for abnormal batch sizes
                                subset = np.random.randint(1, len(batch_i["data"]) - 2)
                        split_1 = subset
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 1:
                        subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        while subset >= len(batch[2]["data"]) - split_1: ## hacky fix for abnormal batch sizes
                            subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                        batch_i = self._take_subset(batch_i, subset)
                    elif idx == 2:
                        subset = len(batch_i["data"]) - split_1 - subset
                        batch_i = self._take_subset(batch_i, subset)
                batch_size += subset

                features, t_loss, losses, metrics = self._shared_step(batch=batch_i, stage="val")

                if save_activations:
                    activations.append(features.detach())

                if self.intersect_only and idx == 0:
                    intersect_batch = self._take_subset(intersect_batch, split_1)
                    features = self._encode(intersect_batch)

                # explicitly call forward to avoid hooks
                for confound in self.domain_classifiers.keys():
                    if domain_preds.get(confound) is None:
                        if self.agg_task_feats:
                            domain_preds[confound] = {}
                        else:
                            domain_preds[confound] = []
                    if domain_targets.get(confound) is None:
                        domain_targets[confound] = []

                    if self.agg_task_feats:
                        for key, feats in features.items():
                            pred_list = domain_preds[confound].get(key)
                            if self.multi_dm_pred:
                                if pred_list is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound][key].forward(feats)] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound][key].forward(feats))
                            else:
                                if pred_list is None:
                                    domain_preds[confound][key] = [self.domain_classifiers[confound].forward(feats)] 
                                else:
                                    domain_preds[confound][key].append(self.domain_classifiers[confound].forward(feats))
                    else:
                        d_pred = self.domain_classifiers[confound].forward(features)
                        domain_preds[confound].append(d_pred)

                    if confound == "age":
                        d_target = self.get_age_targets(batch_i["info"]["subject"], batch_i["info"]["dataset"][0]).int()
                    else:
                        d_target = torch.full((subset,), idx).to(self.device)
                    domain_targets[confound].append(d_target)

                if t_loss is not None:
                    task_loss += t_loss
            
            acc = 0
            for confound in self.domain_classifiers.keys():
                domain_targets[confound] = torch.cat(domain_targets[confound])
                true_domains = domain_targets[confound].detach().cpu().numpy()
                if self.agg_task_feats:
                    accs = []
                    for key, pred_list in domain_preds[confound].items():
                        pred_list = torch.cat(pred_list)
                        pred_list = torch.softmax(pred_list, dim=1)
                        pred_domains = np.argmax(pred_list.detach().cpu().numpy(), axis=1)
                        if confound == "age":
                            pred_domains = pred_domains + 18
                        accs.append(accuracy_score(true_domains, pred_domains))
                    acc += sum(accs) / len(accs)
                else:
                    domain_preds[confound] = torch.cat(domain_preds[confound])
                    domain_preds[confound] = torch.softmax(domain_preds[confound], dim=1)
                    pred_domains = np.argmax(domain_preds[confound].detach().cpu().numpy(), axis=1)
                    if confound == "age":
                        pred_domains = pred_domains + 18
                    acc += accuracy_score(true_domains, pred_domains)

            if save_activations:
                activations = torch.cat(activations).to("cpu")
                label_mapping = {0: 'dataset_1', 1: 'dataset_2'}
                # Convert numerical labels to class names
                label_names = [label_mapping[label.item()] for label in true_domains]
                save_path = Path("/data/engs-pnpl/wolf6942/experiments/MEGalodon/full_run/fullrun_tsne_plots")
                np.save(save_path / f"{self.run_name}_unlearned_activations.npy", activations.numpy())
                np.save(save_path / f"{self.run_name}_unlearned_labels.npy", np.array(label_names))
                print("Saving activations...")
                plot_tsne(activations=activations, labels=label_names, save_dir=save_path, 
                          file_name=f"{self.run_name}_unlearned_tsne.png")

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

    ## Runs single batch through model to generate and save ##
    ## the activations produced by the final layer of the encoder ##
    def get_tsne(self, batch, name=None):
        batch_size = 0
        if self.age_confound:
            age_preds = []
            age_targets = []
        domain_preds = []
        domain_targets = []
        subset = 0
        if self.tsne:
            activations = []
        for idx, batch_i in enumerate(batch):
            if len(batch) == 2:
                if idx == 0:
                    subset = (len(batch_i["data"]) - 1) // 2
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = len(batch_i["data"]) - subset
                    batch_i = self._take_subset(batch_i, subset)
            elif len(batch) == 3:
                if idx == 0 or idx == 1:
                    subset = len(batch_i["data"]) // 3
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 2:
                    start = 2 * len(batch_i["data"]) // 3
                    subset = len(batch_i["data"]) - start
                    batch_i = self._take_subset(batch_i, subset)
            batch_size += subset

            features = self._encode(batch_i)
            activations.append(features.detach())

            if self.age_confound:
                pred_ages = self.domain_classifiers["age"].forward(features)
                pred_ages = torch.softmax(pred_ages, dim=1)
                pred_ages = torch.argmax(pred_ages, dim=1) + 18
                age_preds.append(pred_ages)

                true_ages = self.get_age_targets(batch_i["info"]["subject"], batch_i["info"]["dataset"][0])
                age_targets.append(true_ages)

            d_target = torch.full((subset,), idx).to(self.device)
            domain_targets.append(d_target)

        domain_targets = torch.cat(domain_targets, 0)
        true_domains = domain_targets.cpu().numpy()

        if self.age_confound:
            age_preds = torch.cat(age_preds, 0)
            age_preds = get_age_distribution_labels(ages=age_preds).to(self.device)
            age_preds = F.softmax(age_preds, dim=1)

            age_targets = torch.cat(age_targets, 0)
            age_targets = get_age_distribution_labels(ages=age_targets).to(self.device)
            age_targets = F.softmax(age_targets, dim=1)

            pred_age_dist = age_preds.cpu().numpy()
            true_age_dist = age_targets.cpu().numpy()

        activations = torch.cat(activations).to("cpu")
        label_mapping = {0: 'dataset_1', 1: 'dataset_2', 2: 'dataset_3'}
        # Convert numerical labels to class names
        label_names = [label_mapping[label.item()] for label in true_domains]
        save_path = Path("/data/engs-pnpl/wolf6942/experiments/MEGalodon/MEGalodon-rep-harmonization/subset_tsne_plots")
        if name is not None:
            activs = f"{name}_activations.npy"
            labels = f"{name}_labels.npy"
            if self.age_confound:
                dist_preds = f"{name}_pred_ages.npy"
                true_dist = f"{name}_true_ages.npy"
        else:
            activs = "activations.npy"
            labels = "labels.npy"
        np.save(save_path / activs, activations.numpy())
        np.save(save_path / labels, np.array(label_names))
        if self.age_confound:
            np.save(save_path / dist_preds, pred_age_dist)
            np.save(save_path / true_dist, true_age_dist)
        print("Saving activations...")
        # plot_tsne(activations=activations, labels=label_names, save_dir=save_path, file_name=f"{name}_unlearned_tsne.png")

    def test_step(self, batch, batch_idx): # function not critical for training #
        ## Fine-tuning Mode ##
        if self.finetune:
            features, loss, losses, metrics = self._shared_step(batch=batch, stage="test")
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
                    sync_dist=True,
                )
                self.log_dict(
                    losses,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )
                self.log_dict(
                    metrics,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

            return loss

        ## Pre-training ##
        task_loss = 0
        batch_size = 0
        if self.agg_task_feats:
            domain_preds = {}
        else:
            domain_preds = []
        domain_targets = []
        subset = 0
        split_1 = 0
        for idx, batch_i in enumerate(batch):
            if len(batch) == 2:
                if idx == 0:
                    subset = np.random.randint(1, len(batch_i["data"]) - 1)
                    while subset >= len(batch[1]["data"]): ## hacky fix for abnormal batch sizes
                        subset = np.random.randint(1, len(batch_i["data"]) - 1)
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = len(batch_i["data"]) - subset
                    batch_i = self._take_subset(batch_i, subset)
            elif len(batch) == 3:
                if idx == 0:
                    subset = np.random.randint(1, len(batch_i["data"]) - 2)
                    while subset >= len(batch[1]["data"]) - 2: ## hacky fix for abnormal batch sizes
                        subset = np.random.randint(1, len(batch_i["data"]) - 2)
                    split_1 = subset
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 1:
                    subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                    while subset >= len(batch[2]["data"]) - split_1: ## hacky fix for abnormal batch sizes
                        subset = np.random.randint(1, len(batch_i["data"]) - split_1 - 1)
                    batch_i = self._take_subset(batch_i, subset)
                elif idx == 2:
                    subset = len(batch_i["data"]) - split_1 - subset
                    batch_i = self._take_subset(batch_i, subset)
            batch_size += subset
            features, t_loss, losses, metrics = self._shared_step(batch=batch_i, stage="test")

            if self.agg_task_feats:
                for key, feats in features.items():
                    pred_list = domain_preds.get(key)
                    if self.multi_dm_pred:
                        if pred_list is None:
                            domain_preds[key] = [self.domain_classifiers[key](feats)] 
                        else:
                            domain_preds[key].append(self.domain_classifiers[key](feats))
                    else:
                        if pred_list is None:
                            domain_preds[key] = [self.domain_classifier(feats)] 
                        else:
                            domain_preds[key].append(self.domain_classifier(feats))
            else:
                d_pred = self.domain_classifier(features)
                domain_preds.append(d_pred)

            d_target = torch.full((subset,), idx).to(self.device)
            domain_preds.append(d_pred)
            domain_targets.append(d_target)
            if t_loss is not None:
                task_loss += t_loss

        domain_targets = torch.cat(domain_targets, 0)
        true_domains = domain_targets.detach().cpu().numpy()

        if self.agg_task_feats:
            accs = []
            for key, pred_list in domain_preds.items():
                pred_list = torch.cat(pred_list)
                pred_list = torch.softmax(pred_list, dim=1)
                pred_domains = np.argmax(pred_list.detach().cpu().numpy(), axis=1)
                accs.append(accuracy_score(true_domains, pred_domains))
            acc = sum(accs) / len(accs)
        else:
            domain_preds = torch.cat(domain_preds)
            domain_preds = torch.softmax(domain_preds, dim=1)
            pred_domains = np.argmax(domain_preds.detach().cpu().numpy(), axis=1)
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
    
    # Needed if loading a checkpoint trained with a different optimizer
    def on_load_checkpoint(self, checkpoint):
        if self.clear_optim: # assumes checkpoint was pretrained with adam
            print("CLEARED CHECKPOINT OPTIMS")
            checkpoint["optimizer_states"] = []
        if self.clear_betas: # should be add betas
            for param_group in checkpoint["optimizer_states"]:
                if 'betas' in param_group:
                    del param_group['betas']
                if 'weight_decay' in param_group:
                    del param_group['weight_decay']

    # Needed if loading a checkpoint trained with a different optimizer
    def reset_optims(self):
        print("WORKING")
        for idx, optimizer in enumerate(self.trainer.optimizers):
            for param_group in optimizer.param_groups:
                if idx == 1:
                    param_group["lr"] = self.task_learning_rate
                elif idx == 2:
                    param_group["lr"] = self.conf_learning_rate
                elif idx == 3:
                    param_group["lr"] = self.dm_learning_rate
                # Ensure momentum key is present
                if isinstance(optimizer, torch.optim.SGD):
                    param_group['momentum'] = 0.9 
                    param_group['dampening'] = 0
                    param_group['nesterov'] = True
                    param_group['weight_decay'] = 1e-3

    def configure_optimizers(self):
        encoder_params = list(filter(lambda p: p.requires_grad, self.encoder_models.parameters()))
        predictor_params = list(filter(lambda p: p.requires_grad, self.predictor_models.parameters()))
        domain_classifier_params = []
        for confound in self.domain_classifiers.keys():
            if self.multi_dm_pred:
                for classifier in self.domain_classifiers[confound].values():
                    domain_classifier_params.extend(filter(lambda p: p.requires_grad, classifier.parameters()))
            else:
                domain_classifier_params.extend((filter(lambda p: p.requires_grad, self.domain_classifiers[confound].parameters())))

        if self.finetune:
            return torch.optim.AdamW(
                encoder_params + predictor_params, 
                lr=self.learning_rate
            )
        elif self.sdat:
            base_optim = torch.optim.SGD

            step1_optim = torch.optim.AdamW(
                encoder_params + predictor_params + domain_classifier_params, 
                lr=self.learning_rate
            )

            optim = SAM(encoder_params + predictor_params, base_optim,
                              rho=0.05, adaptive=False, lr=self.task_learning_rate, momentum=0.9,
                              weight_decay=1e-3, nesterov=True)
            dm_optim = SGD(domain_classifier_params, lr=self.dm_learning_rate, 
                                       momentum=0.9, weight_decay=1e-3, nesterov=True)
            
            optim_scheduler = LambdaLR(optim, lambda x: self.task_learning_rate *
                            (1. + 0.001 * float(x)) ** (-0.75))
            dm_scheduler = LambdaLR(
                 dm_optim, lambda x: self.dm_learning_rate * (1. + 0.001 * float(x)) ** (-0.75))
            
            return [
                {'optimizer': step1_optim},
                {'optimizer': optim, 'lr_scheduler': {'scheduler': optim_scheduler, 'interval': 'step', 'frequency': 1}},
                {'optimizer': dm_optim, 'lr_scheduler': {'scheduler': dm_scheduler, 'interval': 'step', 'frequency': 1}}
            ]

        else:
            if self.no_dm_control:
                step1_optim = torch.optim.AdamW(
                    encoder_params + predictor_params, 
                    lr=self.learning_rate
                )
            else:
                step1_optim = torch.optim.AdamW(
                    encoder_params + predictor_params + domain_classifier_params, 
                    lr=self.learning_rate
                )

            optim = torch.optim.Adam(encoder_params + predictor_params, lr=self.task_learning_rate)
            conf_optim = torch.optim.Adam(encoder_params, lr=self.conf_learning_rate)
            if self.sgd:
                dm_optim = SGD(domain_classifier_params, lr=self.dm_learning_rate, 
                                       momentum=0.9, weight_decay=1e-3, nesterov=True)
            else:
                dm_optim = torch.optim.Adam(domain_classifier_params, lr=self.dm_learning_rate)
        
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

    def disable_ssl(self):
        keys = list(self.predictor_models.keys())
        for key in keys:
            if "predictor" in key:
                self.predictor_models.pop(key)

        # Also remove the SSL projector for fine-tuning
        if self.no_proj_encode:
            self.predictor_models["projector"] = nn.Identity()
        else:
            self.encoder_models["projector"] = nn.Identity()

    def disable_classifiers(self):
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
    model = RepHarmonizer(
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
        from dataloaders import MultiDataLoader

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
