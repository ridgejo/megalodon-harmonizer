import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from encodec import EncodecModel

def _make_ch_vqvae(sampling_rate, vq_dim, codebook_size, shared_dim, temporal_dim):

    # encodec_model = EncodecModel.encodec_model_24khz()
    # encodec_model.set_target_bandwidth(12.0) # warning: increase to 12.0 for more accurate reconstructions.
    # temporal_encoder = encodec_model.encoder
    # temporal_decoder = encodec_model.decoder

    # TODO: Update to use SEANet style encoder/decoder setup
    temporal_encoder = TemporalEncoder(
        temporal_dim=temporal_dim,
        hidden_dim=256,
    )
    temporal_decoder = TemporalDecoder(
        temporal_dim=temporal_dim,
        hidden_dim=256,
    )


    quantizer = VectorQuantize(
        dim=vq_dim,
        codebook_size=codebook_size,
        # codebook_dim=16,
        # use_cosine_sim=True,
        # threshold_ema_dead_code=2,
        # kmeans_init=True,
        # kmeans_iters=10,
    )

    return ChVQVAE(
        temporal_encoder=temporal_encoder,
        temporal_decoder=temporal_decoder,
        temporal_dim=temporal_dim,
        quantizer=quantizer,
        sampling_rate=sampling_rate,
        shared_dim=shared_dim,
        vq_dim=vq_dim,
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

class TemporalEncoder(nn.Module):
    def __init__(self, temporal_dim, hidden_dim):
        super(TemporalEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
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
                out_channels=temporal_dim,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # x = [B, C, T] @ 300Hz
        # T1: 10ms shift = 3 time points = 3 stride => 300/3 = 100Hz target
        # T2: 100ms window = 30 time points (3 + 3 + 3 + 3) * 3? => 36
        return self.model(x)

class TemporalDecoder(nn.Module):
    def __init__(self, temporal_dim, hidden_dim):
        super(TemporalDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=temporal_dim,
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
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.model(x)

class ChVQVAE(nn.Module):

    def __init__(self, temporal_encoder, temporal_decoder, temporal_dim, quantizer, sampling_rate, shared_dim, vq_dim):
        super(ChVQVAE, self).__init__()

        self.temporal_encoder = temporal_encoder
        self.temporal_decoder = temporal_decoder
        self.sampling_rate = sampling_rate

        self.spatial_pooling = nn.Conv1d(
            in_channels=shared_dim,
            out_channels=1,
            kernel_size=1,
        )

        self.pre_vq = nn.Conv1d(
            in_channels=temporal_dim,
            out_channels=vq_dim,
            kernel_size=1,
        )

        self.quantizer = quantizer

        self.post_vq = nn.Conv1d(
            in_channels=vq_dim,
            out_channels=temporal_dim,
            kernel_size=1,
        )

        self.spatial_unpooling = nn.Conv1d(
            in_channels=1,
            out_channels=shared_dim,
            kernel_size=1,
        )

        self.act = nn.ELU(alpha=1.0)


    def forward(self, x, dataset_id, subject_id):

        original_x = x.clone()

        # Split channel dimension into segments of size x to avoid memory errors
        sections = x.split(split_size=8, dim=1) # Tuple of split sections [B, C_i, T]
        split_out = []
        for section in sections:
            B, C, T = section.shape
            section = section.flatten(start_dim=0, end_dim=1).unsqueeze(1) # [B * C, 1, T]
            outputs = self.temporal_encoder(section)
            outputs = outputs.permute(0, 2, 1) # [B * C, T, E]
            T, E = outputs.shape[-2:]
            contextual_embeddings = outputs.unflatten(0, (B, C)) # [B, C, T, E]
            split_out.append(contextual_embeddings)
        contextual_embeddings = torch.cat(split_out, dim=1)

        # Pool in spatial dimension and output [B, T, E]
        B, C, T, E = contextual_embeddings.shape
        contextual_embeddings = contextual_embeddings.flatten(start_dim=2, end_dim=3) # [B, C, T * E]
        contextual_embeddings = self.spatial_pooling(contextual_embeddings) # [B, 1, T * E]
        contextual_embeddings = contextual_embeddings.unflatten(2, (T, E)).squeeze(1) # [B, T, E]

        # contextual_embeddings = self.act(contextual_embeddings)

        # # Project embeddings down to VQ dimension
        # codex_embedding = self.pre_vq(contextual_embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        # codex_embedding = self.act(codex_embedding)

        # # Quantize embeddings
        # quantized, codes, commit_loss = self.quantizer(codex_embedding)

        # # Project quantized embeddings up to transformer dimension
        # z = self.post_vq(quantized.permute(0, 2, 1)).permute(0, 2, 1)

        # z = self.act(z)

        z = contextual_embeddings

        # Unpool in spatial dimension
        z = z.unsqueeze(1) # [B, 1, T, E]
        B, C, T, E = z.shape
        z = z.flatten(start_dim=2, end_dim=3) # [B, 1, T * E]
        z = self.spatial_unpooling(z) # [B, C, T * E]
        z = z.unflatten(2, (T, E)) # [B, C, T, E]

        # z = self.act(z)


        # Split channel dimension into segments of size x to avoid memory errors
        sections = z.split(split_size=8, dim=1)
        segments = []
        for section in sections:
            B, C, T, E = section.shape
            section = section.permute(0, 1, 3, 2) # [B, C, E, T]
            section = section.flatten(start_dim=0, end_dim=1) # [B * C, E, T]
            section = self.temporal_decoder(section) # [B * C, 1, T]
            T = section.shape[-1]
            section = section.squeeze(1) # [B * C, T]
            output_waves = section.unflatten(0, (B, C)) # [B, C, T]
            segments.append(output_waves)
        output_waves = torch.cat(segments, dim=1)

        # Compute losses
        commit_loss = 0.0
        recon_loss = F.mse_loss(original_x, output_waves)
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

        return output_waves, loss

if __name__ == "__main__":

    import torch

    model = _make_ch_vqvae(
        shared_dim=20,
        temporal_dim=512, # Output dimension of encodec model
        sampling_rate=300,
        vq_dim=512,
        codebook_size=1024
    ).cuda()

    x = torch.randn(1, 20, 90).cuda() # [B, C, T]
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    out_wave, loss = model(x, dataset_id, subject_id)

    assert out_wave.shape == x.shape