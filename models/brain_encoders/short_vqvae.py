"""Short VQ-VAE. Designed to produce embeddings with 100ms receptive fields and 10ms shift (90ms overlap)."""

import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock


def _make_short_vqvae(
    vq_dim,
    codebook_size,
    shared_dim,
    hidden_dim,
    dataset_sizes,
    subject_ids,
    use_sub_block,
    use_data_block,
):
    encoder = ShortEncoder(
        vq_dim=vq_dim,
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
    )
    decoder = ShortDecoder(
        vq_dim=vq_dim,
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
    )

    # encoder = SEANetBrainEncoder(
    #     channels=shared_dim,
    #     ratios=[3, 1, 1],
    #     conv_channels=[512, 512, 512, 1024],
    #     dimension=vq_dim,
    #     causal=True,
    #     n_residual_layers=1,
    # )
    # decoder = SEANetBrainDecoder(
    #     channels=shared_dim,
    #     ratios=[3, 1, 1],
    #     conv_channels=[512, 512, 512, 1024],
    #     dimension=vq_dim,
    #     causal=True,
    #     n_residual_layers=1,
    # )

    quantizer = VectorQuantize(
        dim=vq_dim,
        codebook_size=codebook_size,
        codebook_dim=16,
        use_cosine_sim=True,
        threshold_ema_dead_code=2,
        kmeans_init=True,
        kmeans_iters=10,
    )

    # quantizer = GroupedResidualVQ(
    #     dim=vq_dim,
    #     num_quantizers=8,
    #     groups=2,
    #     codebook_size=codebook_size,
    #     use_cosine_sim = True,
    #     # replace codes that have EMA cluster size less than 2
    #     threshold_ema_dead_code = 2,
    #     kmeans_init = True,   # set to True
    #     kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
    # )

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

    return ShortVQVAE(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        dataset_layer=dataset_layer,
        subject_block=subject_block,
    )


class ResnetBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(ResnetBlock, self).__init__()

        hidden = dim // 2

        self.block = nn.Sequential(
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=dim,
                out_channels=hidden,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=hidden,
                out_channels=dim,
                kernel_size=1,
            ),
        )

        self.shortcut = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ShortEncoder(nn.Module):
    def __init__(self, vq_dim, shared_dim, hidden_dim):
        super(ShortEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=shared_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),  # rf: 6
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),  # 12
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),  # 18
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),  # 24
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=3
            ),  # 30 (+ downsample)
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=vq_dim,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # x = [B, C, T] @ 300Hz
        # T1: 10ms shift = 3 time points = 3 stride => 300/3 = 100Hz target
        # T2: 100ms window = 30 time points (3 + 3 + 3 + 3) * 3? => 36
        return self.model(x)


class ShortDecoder(nn.Module):
    def __init__(self, vq_dim, shared_dim, hidden_dim):
        super(ShortDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=vq_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            nn.ELU(alpha=1.0),
            nn.ConvTranspose1d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=3
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=7,
            ),
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=shared_dim,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.model(x)


class ShortVQVAE(nn.Module):
    def __init__(self, encoder, decoder, quantizer, dataset_layer, subject_block):
        super(ShortVQVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.dataset_layer = dataset_layer
        self.subject_block = subject_block

    def encode(self, x, dataset_id, subject_id):
        x = self.dataset_layer(x, dataset_id)
        x = self.subject_block(x, subject_id)
        x = self.encoder(x)  # [B, C, T]
        quantized, codes, commit_loss = self.quantize(x)
        return quantized, codes, commit_loss

    def quantize(self, z):
        z = z.permute(0, 2, 1)
        quantized, codes, commit_loss = self.quantizer(
            z  # [B, T, C]
        )
        quantized = quantized.permute(0, 2, 1)
        return quantized, codes, commit_loss

    def decode(self, quantized, dataset_id, subject_id):
        x_hat = self.decoder(quantized)
        x_hat = self.subject_block.decode(x_hat, subject_id)
        x_hat = self.dataset_layer.decode(x_hat, dataset_id)
        return x_hat

    def decode_codes(self, codes):
        quantized = self.quantizer.indices_to_codes(codes)
        return self.decode(quantized)

    def forward(self, x, dataset_id, subject_id):
        quantized, _, commit_loss = self.encode(x, dataset_id, subject_id)
        x_hat = self.decode(quantized, dataset_id, subject_id)

        commit_loss = commit_loss.mean()
        recon_loss = F.mse_loss(x, x_hat)
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

        return x_hat, loss


if __name__ == "__main__":
    import torch

    short_vqvae = _make_short_vqvae(
        vq_dim=64,
        codebook_size=1024,
        shared_dim=128,
        hidden_dim=768,
        dataset_sizes={
            "TestDataset": 269,
        },
        subject_ids=["TestSubject"],
    )

    x = torch.randn(8, 269, 150)
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    x_hat, loss = short_vqvae(x, dataset_id, subject_id)

    print("Loss", loss)

    assert x.shape == x_hat.shape

    print("PASS. Input and output are the same size!")
