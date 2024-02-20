import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize, GroupedResidualVQ
from encodec import EncodecModel

from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock
from models.brain_encoders.seanet.seanet import SEANetBrainEncoder, SEANetBrainDecoder

def _make_seanet_vqvae(vq_dim, codebook_size, shared_dim, ratios, conv_channels, dataset_sizes,
    subject_ids,
    use_sub_block,
    use_data_block,
    rvq=False):

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

    return SEANetVQVAE(
        dataset_layer=dataset_layer,
        subject_block=subject_block,
        temporal_encoder=temporal_encoder,
        temporal_decoder=temporal_decoder,
        quantizer=quantizer,
        shared_dim=shared_dim,
        vq_dim=vq_dim,
    )

class SEANetVQVAE(nn.Module):

    def __init__(self, dataset_layer, subject_block, temporal_encoder, temporal_decoder, quantizer, shared_dim, vq_dim):
        super(SEANetVQVAE, self).__init__()

        self.dataset_layer = dataset_layer
        self.subject_block = subject_block

        self.temporal_encoder = temporal_encoder
        self.temporal_decoder = temporal_decoder

        self.quantizer = quantizer
    
    def encode(self, x, dataset_id, subject_id):

        x = self.dataset_layer(x, dataset_id)
        x = self.subject_block(x, subject_id)

        x = self.temporal_encoder(x) # [B, C, T] -> [B, E, T @ 75Hz]

        # Vector quantization
        B, E, T = x.shape
        quantized, codes, commit_loss = self.quantizer(x.permute(0, 2, 1))
        quantized = quantized.permute(0, 2, 1)

        return quantized, codes, commit_loss

    def decode(self, x, dataset_id, subject_id):

        # Expand in temporal dimension
        x = self.temporal_decoder(x) # [B, E, T @ 75] -> [B, C, T]

        x = self.subject_block.decode(x, subject_id)
        x = self.dataset_layer.decode(x, dataset_id)

        return x

    def forward(self, x, dataset_id, subject_id):

        original_x = x.clone() # [B, S, T]

        quantized, codes, commit_loss = self.encode(x, dataset_id, subject_id)
        commit_loss = commit_loss.sum() # In case of RVQ

        x = self.decode(quantized, dataset_id, subject_id)
        
        # Compute losses
        recon_loss = F.mse_loss(original_x, x)
        loss = {
            "loss": recon_loss + commit_loss,
            "commit_loss": commit_loss,
            "recon_loss": recon_loss,
            f"D_{dataset_id}": 1,
            f"D_{dataset_id}_loss": recon_loss + commit_loss,
            f"D_{dataset_id}_commit_loss": commit_loss,
            f"D_{dataset_id}_recon_loss": recon_loss,
            f"S_{dataset_id}_{subject_id}": 1,
            f"S_{dataset_id}_{subject_id}_loss": recon_loss + commit_loss,
            f"S_{dataset_id}_{subject_id}_commit_loss": commit_loss,
            f"S_{dataset_id}_{subject_id}_recon_loss": recon_loss,
        }

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