import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from encodec import EncodecModel

from models.dataset_layer import DatasetLayer
from models.subject_block import SubjectBlock

def _make_ch_vqvae(sampling_rate, vq_dim, codebook_size, shared_dim, temporal_dim, hidden_dim, dataset_sizes,
    subject_ids,
    use_sub_block,
    use_data_block,):

    # encodec_model = EncodecModel.encodec_model_24khz()
    # encodec_model.set_target_bandwidth(12.0) # warning: increase to 12.0 for more accurate reconstructions.
    # temporal_encoder = encodec_model.encoder
    # temporal_decoder = encodec_model.decoder

    # TODO: Update to use SEANet style encoder/decoder setup
    temporal_encoder = TemporalEncoder(
        ch_in_dim=1,
        hidden_dim=hidden_dim,
        ch_out_dim=temporal_dim,
    )
    temporal_decoder = TemporalDecoder(
        ch_in_dim=1,
        hidden_dim=hidden_dim,
        ch_out_dim=temporal_dim,
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

    return ChVQVAE(
        dataset_layer=dataset_layer,
        subject_block=subject_block,
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
            nn.Conv2d(
                in_channels=dim,
                out_channels=hidden,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=hidden,
                out_channels=dim,
                kernel_size=1,
            ),
        )

        self.shortcut = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class TemporalEncoder(nn.Module):
    def __init__(self, ch_in_dim, hidden_dim, ch_out_dim):
        super(TemporalEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),  # rf: 6
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),  # 12
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),  # 18
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),  # 24
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 3), stride=(1, 3),
            ),  # 30 (+ downsample)
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=ch_out_dim,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # x = [B, C, T] @ 300Hz
        # T1: 10ms shift = 3 time points = 3 stride => 300/3 = 100Hz target
        # T2: 100ms window = 30 time points (3 + 3 + 3 + 3) * 3? => 36
        return self.model(x)

class TemporalDecoder(nn.Module):
    def __init__(self, ch_in_dim, hidden_dim, ch_out_dim):
        super(TemporalDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_out_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            nn.ELU(alpha=1.0),
            nn.ConvTranspose2d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 3), stride=(1, 3),
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),
            ResnetBlock(
                dim=hidden_dim,
                kernel_size=(1, 7),
            ),
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=ch_in_dim,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.model(x)

class ChVQVAE(nn.Module):

    def __init__(self, dataset_layer, subject_block, temporal_encoder, temporal_decoder, temporal_dim, quantizer, sampling_rate, shared_dim, vq_dim):
        super(ChVQVAE, self).__init__()

        self.dataset_layer = dataset_layer
        self.subject_block = subject_block

        self.temporal_encoder = temporal_encoder
        self.temporal_decoder = temporal_decoder

        ch_dim = temporal_dim

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_dim,
                out_channels=ch_dim * 4,
                kernel_size=1,
            ),
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=ch_dim * 4,
                out_channels=ch_dim * 4,
                kernel_size=(30, 1),
                dilation=(4, 1),
                stride=(20, 1), # 300 -> 10 channels
            ),
            nn.ELU(alpha=1.0),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=ch_dim * 4,
                out_channels=ch_dim * 4,
                stride=(5, 1), # Take 1 -> 10
                kernel_size=(10, 1),
            ),
            nn.ELU(alpha=1.0),
            nn.ConvTranspose2d(
                in_channels=ch_dim * 4,
                out_channels=ch_dim * 4,
                kernel_size=(30, 1),
                dilation=(4, 1),
                stride=(20, 1), # 10 -> 300 channels
                output_padding=(3, 0),
            ),
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=ch_dim * 4,
                out_channels=ch_dim,
                kernel_size=1,
            ),
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

    def forward(self, x, dataset_id, subject_id):

        original_x = x.clone() # [B, S, T]

        x = self.dataset_layer(x, dataset_id)
        x = self.subject_block(x, subject_id)

        # Expand to create channel (embedding) dimension
        x = x.unsqueeze(1) # [B, C, S, T]

        # Apply temporal encoder [B, C, S, T @ 250Hz] -> [B, C, S, T @ ~62Hz]
        x = self.temporal_encoder(x)

        # Apply spatial pooling layers
        x = self.spatial_encoder(x)
        x = x.mean(dim=2) # Average pool over what's left in the spatial dimension

        # Expand in spatial dimension
        x = x.unsqueeze(2)
        x = self.spatial_decoder(x)

        # Expand in temporal dimension
        # warning: not equal to spatial dimension
        x = self.temporal_decoder(x)
        x = x.squeeze(1)

        x = self.subject_block.decode(x, subject_id)
        x = self.dataset_layer.decode(x, dataset_id)

        # Compute losses
        commit_loss = 0.0
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

    model = _make_ch_vqvae(
        shared_dim=300,
        temporal_dim=512, # Output dimension of encodec model
        sampling_rate=300,
        vq_dim=512,
        codebook_size=1024
    ).cuda()

    x = torch.randn(1, 300, 90).cuda() # [B, C, T]
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    out_wave, loss = model(x, dataset_id, subject_id)

    assert out_wave.shape == x.shape