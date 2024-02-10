import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize, FSQ
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
        ch_in_dim=temporal_dim,
        hidden_dim=hidden_dim,
        ch_out_dim=temporal_dim,
    )
    temporal_decoder = TemporalDecoder(
        ch_in_dim=temporal_dim,
        hidden_dim=hidden_dim,
        ch_out_dim=temporal_dim,
    )

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
            # nn.Conv2d(
            #     in_channels=hidden_dim,
            #     out_channels=ch_out_dim,
            #     kernel_size=1,
            # ),
            # nn.ELU(alpha=1.0),
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
            # nn.Conv2d(
            #     in_channels=ch_out_dim,
            #     out_channels=hidden_dim,
            #     kernel_size=1,
            # ),
            # nn.ELU(alpha=1.0),
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

        # 300 -> 29 channels
        self.spatial_fuse30 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=ch_dim,
                kernel_size=1,
            ),
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=ch_dim,
                out_channels=ch_dim,
                kernel_size=(60, 1), # 20, 1
                stride=(30, 1), # 10, 1
            ),
            nn.ELU(alpha=1.0),
        )

        # # 29 -> 1
        # self.spatial_fuse1 = nn.Sequential(
        #     # nn.Conv2d(
        #     #     in_channels=ch_dim * 2,
        #     #     out_channels=ch_dim * 2,
        #     #     kernel_size=1,
        #     # ),
        #     # nn.ELU(alpha=1.0),
        #     nn.Conv2d(
        #         in_channels=ch_dim * 2,
        #         out_channels=ch_dim * 2,
        #         kernel_size=(29, 1),
        #     ),
        #     nn.ELU(alpha=1.0),
        # )

        # # 1 -> 29
        # self.spatial_unfuse1 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=ch_dim * 2,
        #         out_channels=ch_dim * 2,
        #         kernel_size=(29, 1),
        #     ),
        #     nn.ELU(alpha=1.0),
        #     # nn.Conv2d(
        #     #     in_channels=ch_dim * 2,
        #     #     out_channels=ch_dim * 2,
        #     #     kernel_size=1,
        #     # ),
        #     # nn.ELU(alpha=1.0),
        # )

        # 29 -> 300
        self.spatial_unfuse30 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=ch_dim,
                out_channels=ch_dim,
                kernel_size=(60, 1),
                stride=(30, 1),
            ),
            nn.ELU(alpha=1.0),
            nn.Conv2d(
                in_channels=ch_dim,
                out_channels=1,
                kernel_size=1,
            ),
            nn.ELU(alpha=1.0),
        )

        self.pre_vq = nn.Conv2d(
            in_channels=ch_dim * 2,
            out_channels=vq_dim,
            kernel_size=1,
        )

        self.post_vq = nn.Conv2d(
            in_channels=vq_dim,
            out_channels=ch_dim * 2,
            kernel_size=1,
        )

        self.quantizer = quantizer

    def forward(self, x, dataset_id, subject_id):

        original_x = x.clone() # [B, S, T]

        x = self.dataset_layer(x, dataset_id)
        x = self.subject_block(x, subject_id)

        # Expand to create channel (embedding) dimension (C = 1)
        x = x.unsqueeze(1) # [B, C, S, T]

        x = self.spatial_fuse30(x) # S = 300 -> S = 29 (C = 128)

        # Apply temporal encoder [B, C, S, T @ 250Hz] -> [B, C, S, T @ ~62Hz]
        x = self.temporal_encoder(x) # (C = 256)

        # x = self.spatial_fuse1(x) # S = 29 -> S = 1 (C = 512)

        # # Vector quantization
        x = self.pre_vq(x)
        B, C, S, T = x.shape
        x = x.flatten(start_dim=2, end_dim=3)

        quantized, codes, commit_loss = self.quantizer(x.permute(0, 2, 1))

        x = quantized.permute(0, 2, 1)
        x = x.unflatten(2, (S, T))
        x = self.post_vq(x)

        # Expand in spatial dimension
        # x = self.spatial_unfuse1(x) # S = 1 -> S = 29

        # Expand in temporal dimension
        x = self.temporal_decoder(x)

        x = self.spatial_unfuse30(x) # S = 29 -> S = 300
        x = x.squeeze(1)

        x = self.subject_block.decode(x, subject_id)
        x = self.dataset_layer.decode(x, dataset_id)

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

    model = _make_ch_vqvae(
        shared_dim=300,
        temporal_dim=128, # Output dimension of encodec model
        hidden_dim=256,
        sampling_rate=300,
        vq_dim=64,
        codebook_size=1024,
        dataset_sizes={
            "TestDataset": 269,
        },
        subject_ids=["TestSubject"],
        use_sub_block=True,
        use_data_block=True,
    ).cuda()

    x = torch.randn(1, 269, 90).cuda() # [B, C, T]
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    out_wave, loss = model(x, dataset_id, subject_id)

    assert out_wave.shape == x.shape