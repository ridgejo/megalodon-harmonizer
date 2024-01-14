"""Short VQ-VAE. Designed to produce embeddings with 100ms receptive fields and 10ms shift (90ms overlap)."""

import torch.nn as nn
import torch.nn.functional as F
from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock
from vector_quantize_pytorch import VectorQuantize

def _make_short_vqvae(vq_dim, codebook_size, shared_dim, hidden_dim, dataset_sizes, subject_ids):

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

    quantizer = VectorQuantize(
        dim=vq_dim,
        codebook_size=codebook_size,
        codebook_dim = 16,
        use_cosine_sim = True,
        threshold_ema_dead_code = 2,
        kmeans_init = True,
        kmeans_iters = 10,
    )

    dataset_layer = DatasetLayer(
        dataset_sizes=dataset_sizes,
        shared_dim=shared_dim
    )

    subject_block = SubjectBlock(
        subject_ids=subject_ids,
        in_channels=shared_dim,
        out_channels=shared_dim,
    )

    return ShortVQVAE(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        dataset_layer=dataset_layer,
        subject_block=subject_block,
    )

class ShortEncoder(nn.Module):
    def __init__(self, vq_dim, shared_dim, hidden_dim):
        super(ShortEncoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=shared_dim, out_channels=hidden_dim, kernel_size=7, padding="same"), # rf: 6
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding="same"), # 12
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding="same"), # 18
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding="same"), # 24
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=vq_dim, kernel_size=3, stride=3), # 30
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
            nn.ConvTranspose1d(in_channels=vq_dim, out_channels=hidden_dim, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=shared_dim, kernel_size=7, padding="same"),
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
        x = self.encoder(x) # [B, C, T]
        quantized, codes, commit_loss = self.quantize(x)
        return quantized, codes, commit_loss

    def quantize(self, z):
        z = z.permute(0, 2, 1)
        quantized, codes, commit_loss = self.quantizer(
            z # [B, T, C]
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

        recon_loss = F.mse_loss(x, x_hat)
        loss = {
            "loss": recon_loss + commit_loss,
            "commit_loss": commit_loss,
            "recon_loss": recon_loss,
        }

        return x_hat, loss

if __name__ == "__main__":

    import torch

    short_vqvae = _make_short_vqvae(
        vq_dim=128,
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