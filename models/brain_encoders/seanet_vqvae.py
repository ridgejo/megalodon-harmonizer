import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize, GroupedResidualVQ
from encodec import EncodecModel

from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock
from models.brain_encoders.seanet.seanet import SEANetBrainEncoder, SEANetBrainDecoder

from models.mlp_seq import _make_mlp_seq

class ProjectorMLP(nn.Module):

    # Use as projector in SSL

    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ProjectorMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
        )
    
    def forward(self, x):
        return self.model(x)

        

def _make_seanet_vqvae(vq_dim, codebook_size, shared_dim, ratios, conv_channels, dataset_sizes,
    subject_ids,
    use_sub_block,
    use_data_block,
    rvq=False,
    use_transformer=False,
    vad_scale=1.0,
    objective="recon"):

    # TODO: Update to use SEANet style encoder/decoder setup

    temporal_encoder = SEANetBrainEncoder(
        channels=shared_dim,
        conv_channels=conv_channels, #[128, 256, 512, 1024], # 250
        ratios=ratios, # warning: sampling rate must be divisible by this
        dimension=vq_dim,
        causal=True,
    )

    temporal_decoder = SEANetBrainDecoder(
        channels=shared_dim,
        conv_channels=conv_channels, #[128, 256, 512, 1024],
        ratios=ratios,
        dimension=vq_dim,
        causal=True,
    )

    if rvq:
        quantizer = GroupedResidualVQ(
            dim=vq_dim,
            num_quantizers=8,
            groups=2,
            codebook_size=codebook_size,
        )
    else:
        quantizer = VectorQuantize(
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=16,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
            kmeans_init=True,
            kmeans_iters=10,
        )

    dataset_layer = DatasetLayer(
        dataset_sizes=dataset_sizes,
        shared_dim=shared_dim,
        use_data_block=use_data_block,
    )

    subject_block = SubjectBlock(
        subject_ids=subject_ids,
        in_channels=shared_dim,
        out_channels=shared_dim,
        use_sub_block=use_sub_block,
    )

    vad_classifier = _make_mlp_seq(
        dataset_sizes=dataset_sizes,
        use_data_block=False,
        subject_ids=subject_ids,
        use_sub_block=True,#"sub_block" in config["model"]["mlp"],
        feature_dim=vq_dim,
        hidden_dim=256,
        output_classes=2,
    )

    return SEANetVQVAE(
        dataset_layer=dataset_layer,
        subject_block=subject_block,
        temporal_encoder=temporal_encoder,
        temporal_decoder=temporal_decoder,
        quantizer=quantizer,
        shared_dim=shared_dim,
        vq_dim=vq_dim,
        use_transformer=use_transformer,
        vad_classifier=vad_classifier,
        vad_scale=vad_scale,
        objective=objective,
    )

class SEANetVQVAE(nn.Module):

    def __init__(self, dataset_layer, subject_block, temporal_encoder, temporal_decoder, quantizer, shared_dim, vq_dim, use_transformer, vad_classifier, vad_scale, objective):
        super(SEANetVQVAE, self).__init__()

        self.dataset_layer = dataset_layer
        self.subject_block = subject_block

        self.temporal_encoder = temporal_encoder
        self.temporal_decoder = temporal_decoder

        self.use_transformer = use_transformer

        self.vad_scale = vad_scale
        self.objective = objective

        if self.use_transformer:
            # aim: build strong contextual representation in latent space.
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    batch_first=True,
                    d_model=vq_dim,
                    nhead=8,
                ),
                num_layers=4, # 2 worked well. 6 failed.
            )
            self.transformer_decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    batch_first=True,
                    d_model=vq_dim,
                    nhead=8,
                ),
                num_layers=4,
            )

        self.quantizer = quantizer

        # Set up projectors
        self.dec_projection = ProjectorMLP(in_dim=vq_dim, out_dim=vq_dim, hidden_dim=256)

        self.vad_classifier = vad_classifier
    
    def encode(self, x, dataset_id, subject_id):

        x = self.dataset_layer(x, dataset_id)
        x = self.subject_block(x, subject_id)

        x = self.temporal_encoder(x) # [B, C, T] -> [B, E, T @ 75Hz]

        if self.use_transformer:
            x = self.transformer_encoder(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Vector quantization
        B, E, T = x.shape
        quantized, codes, commit_loss = self.quantizer(x.permute(0, 2, 1))
        quantized = quantized.permute(0, 2, 1)

        return quantized, codes, commit_loss.sum() # In case of RVQ

    def decode(self, x, dataset_id, subject_id):

        if self.use_transformer:
            x = self.transformer_decoder(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Expand in temporal dimension
        x = self.temporal_decoder(x) # [B, E, T @ 75] -> [B, C, T]

        x = self.subject_block.decode(x, subject_id)
        x = self.dataset_layer.decode(x, dataset_id)

        return x

    def forward(self, x, dataset_id, subject_id, vad_labels=None):

        if self.objective == "autoregressive":
            original_x = x.clone()
            T = x.shape[-1]
            assert T % 2 == 0, "Sequence length must be even for autoregressive VQ-VAE objective"
            x_t0 = x[:, :, : T // 2]
            x_t1 = x[:, :, T // 2 :]
        elif self.objective == "recon":
            original_x = x.clone() # [B, S, T]
            x_t0 = x
            x_t1 = x
        else:
            raise ValueError(f"Unknown objective: {objective}")

        quantized, codes, commit_loss = self.encode(x_t0, dataset_id, subject_id)

        # Reconstruction objective
        B, _, T = quantized.shape
        dec_projection = self.dec_projection(
            quantized.permute(0, 2, 1).flatten(start_dim=0, end_dim=1)
        ).unflatten(dim=0,sizes=(B, T)).permute(0, 2, 1)
        x = self.decode(dec_projection, dataset_id, subject_id)

        decoder_loss = F.mse_loss(x, x_t1) # Automatically accounts for autoregressive vs reconstruction

        # VAD objective (only computed when labels are available)
        if vad_labels is not None:
            vad_labels = F.interpolate(vad_labels.unsqueeze(1), size=quantized.shape[-1]).squeeze(1)
            vad_prediction, vad_loss = self.vad_classifier(quantized, vad_labels, dataset_id, subject_id)
            vad_loss = vad_loss["loss"]
        else:
            vad_loss = 0.0
        
        # Compute losses
        total_loss = decoder_loss + commit_loss + self.vad_scale * vad_loss
        loss = {
            "loss": total_loss,
            "commit_loss": commit_loss,
            "decoder_loss": decoder_loss,
            f"D_{dataset_id}": 1,
            f"D_{dataset_id}_loss": total_loss,
            f"D_{dataset_id}_commit_loss": commit_loss,
            f"D_{dataset_id}_decoder_loss": decoder_loss,
            f"S_{dataset_id}_{subject_id}": 1,
            f"S_{dataset_id}_{subject_id}_loss": total_loss,
            f"S_{dataset_id}_{subject_id}_commit_loss": commit_loss,
            f"S_{dataset_id}_{subject_id}_decoder_loss": decoder_loss,
        }

        if vad_labels is not None:
            loss = dict(loss, **{
                "vad_loss": self.vad_scale * vad_loss,
                f"D_{dataset_id}_vad_loss": vad_loss,
                f"S_{dataset_id}_{subject_id}_vad_loss": vad_loss,
            })

        return x, loss

if __name__ == "__main__":

    import torch

    model = _make_seanet_vqvae(
        shared_dim=300,
        vq_dim=64,
        codebook_size=1024,
        ratios=[2, 1, 1],
        conv_channels=[256, 512, 512, 512],
        dataset_sizes={
            "TestDataset": 269,
        },
        subject_ids=["TestSubject"],
        use_sub_block=True,
        use_data_block=True,
    ).cuda()

    x = torch.randn(1, 269, 250 * 3).cuda() # [B, C, T @ 250Hz 3 seconds]
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    out_wave, loss = model(x, dataset_id, subject_id)

    assert out_wave.shape == x.shape