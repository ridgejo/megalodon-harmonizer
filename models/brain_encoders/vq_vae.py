from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from brain_tokenizers.seanet.seanet import SEANetBrainDecoder, SEANetBrainEncoder
from vector_quantize_pytorch import GroupedResidualVQ, VectorQuantize


def _make_cortex_vqvae_from_config(config):
    return _make_cortex_vqvae(
        brain_channels=config["model"]["brain_channels"],
        ratios=config["model"]["ratios"],
        conv_channels=config["model"]["conv_channels"],
        dimension=config["model"]["dimension"],
        n_residual_layers=config["model"]["n_residual_layers"],
        codebook_size=config["model"]["codebook_size"],
        config=config,
    )


def _make_cortex_vqvae(
    brain_channels: int,
    ratios: List[int],
    conv_channels: List[int],
    dimension: int,
    n_residual_layers: int,
    codebook_size: int,
    config,
):
    encoder = SEANetBrainEncoder(
        channels=brain_channels,
        ratios=ratios,
        conv_channels=conv_channels,
        dimension=dimension,
        causal=True,
        n_residual_layers=n_residual_layers,
    )

    decoder = SEANetBrainDecoder(
        channels=brain_channels,
        ratios=ratios,
        conv_channels=conv_channels,
        dimension=dimension,
        causal=True,
        n_residual_layers=n_residual_layers,
    )

    quantizer = VectorQuantize(
        dim=dimension,
        codebook_size=codebook_size,
        codebook_dim=16,  # Down-projecting can help!
        use_cosine_sim=True,
        # replace codes that have EMA cluster size less than 2
        threshold_ema_dead_code=2,
        # decay=0.8,
        # commitment_weight=1.0,
        kmeans_init=True,  # set to True
        kmeans_iters=10,  # number of kmeans iterations to calculate the centroids for the codebook on init
    )

    # quantizer = FSQ(
    #     levels=[8, 8, 8, 6, 5] # Equivalent to 16384 codebook
    # )

    quantizer = GroupedResidualVQ(
        dim=dimension,
        num_quantizers=32,
        groups=2,
        codebook_size=codebook_size,
        use_cosine_sim=True,
        # replace codes that have EMA cluster size less than 2
        threshold_ema_dead_code=2,
        kmeans_init=True,  # set to True
        kmeans_iters=10,  # number of kmeans iterations to calculate the centroids for the codebook on init
    )

    return CortexVQVAE(encoder, decoder, quantizer, config)


class CortexVQVAE(nn.Module):
    def __init__(self, encoder, decoder, quantizer, config):
        super(CortexVQVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.config = config

    def forward(self, brain_wave):
        """brain_waves is a [Batch, Sequence len (T), Channels] raw brain wave input."""
        encoded_embs = self.encoder(brain_wave)  # [B, C, T]
        quantized_embs, codes, commit_loss = self.quantizer(
            encoded_embs.permute(0, 2, 1)  # [B, T, C]
        )
        decoded_wave = self.decoder(quantized_embs.permute(0, 2, 1))  # [B, C, T]
        commit_loss = commit_loss.mean()

        recon_loss = F.mse_loss(brain_wave, decoded_wave)
        self.loss = {
            "loss": commit_loss + recon_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
        }

        return decoded_wave

    def encode(self, brain_wave):
        """Encodes and quantizes brain waves into a frame."""
        encoded_embs = self.encoder(brain_wave)
        quantized_embs, codes, commit_loss = self.quantizer(
            encoded_embs.permute(0, 2, 1)
        )
        return ([codes], [quantized_embs.permute(0, 2, 1)])

    def decode(self, codes):
        return self.quantizer.get_output_from_indices(codes)

    def losses(self, x, x_hat, config):
        return self.loss

    def get_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["lr"])

    def save(self, folder, epoch):
        torch.save(self.state_dict(), folder / f"{epoch + 1}.pt")
