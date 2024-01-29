import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from encodec import EncodecModel

def _make_attention_vqvae(sampling_rate, vq_dim, codebook_size, shared_dim, temporal_dim, transformer_dim):

    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0) # warning: increase to 12.0 for more accurate reconstructions.
    temporal_encoder = encodec_model.encoder
    temporal_decoder = encodec_model.decoder

    quantizer = VectorQuantize(
        dim=vq_dim,
        codebook_size=codebook_size,
        codebook_dim=16,
        use_cosine_sim=True,
        threshold_ema_dead_code=2,
        kmeans_init=True,
        kmeans_iters=10,
    )

    return AttentionVQVAE(
        temporal_encoder=temporal_encoder,
        temporal_decoder=temporal_decoder,
        temporal_dim=temporal_dim,
        quantizer=quantizer,
        sampling_rate=sampling_rate,
        shared_dim=shared_dim,
        transformer_dim=transformer_dim,
        vq_dim=vq_dim,
    )



class AttentionVQVAE(nn.Module):

    def __init__(self, temporal_encoder, temporal_decoder, temporal_dim, quantizer, sampling_rate, shared_dim, transformer_dim, vq_dim):
        super(AttentionVQVAE, self).__init__()

        self.temporal_encoder = temporal_encoder
        self.temporal_decoder = temporal_decoder
        self.sampling_rate = sampling_rate

        self.spatial_pooling = nn.Conv1d(
            in_channels=shared_dim,
            out_channels=1,
            kernel_size=1,
        )

        self.project_transformer = nn.Conv1d(
            in_channels=temporal_dim,
            out_channels=transformer_dim,
            kernel_size=1,
        )

        self.codex_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.pre_vq = nn.Conv1d(
            in_channels=transformer_dim,
            out_channels=vq_dim,
            kernel_size=1,
        )

        self.quantizer = quantizer

        self.post_vq = nn.Conv1d(
            in_channels=vq_dim,
            out_channels=transformer_dim,
            kernel_size=1,
        )

        self.codex_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.project_temporal = nn.Conv1d(
            in_channels=transformer_dim,
            out_channels=temporal_dim,
            kernel_size=1,
        )

        self.spatial_unpooling = nn.Conv1d(
            in_channels=1,
            out_channels=shared_dim,
            kernel_size=1,
        )

        self.embedding_pooling = nn.Conv1d(
            in_channels=transformer_dim,
            out_channels=1,
            kernel_size=1,
        )


    def forward(self, x, dataset_id, subject_id):

        original_x = x.clone()

        original_samples = x.shape[-1]

        with torch.no_grad():
            # Encode all channels individually in time with Encodec embeddings

            # Upsample signal to encodec sampling rate (24kHz)
            x = F.interpolate(x, scale_factor=24000 // self.sampling_rate)

            x = x.permute(1, 0, 2) # [C, B, T]

            original_size = x.shape[-1]

            contextual_embeddings = []
            for batched_channel in x:
                # warning: assumes zero mean and unit variance
                batched_channel = batched_channel.unsqueeze(1)
                outputs = self.temporal_encoder(batched_channel)
                outputs = outputs.permute(0, 2, 1) # [B, T, E]
                contextual_embeddings.append(outputs)
            
            contextual_embeddings = torch.stack(contextual_embeddings) # [C, B, T, E]
            contextual_embeddings = contextual_embeddings.permute(1, 0, 2, 3) # [B, C, T, E]

        # Pool in spatial dimension and output [B, T, E]
        B, C, T, E = contextual_embeddings.shape
        contextual_embeddings = contextual_embeddings.view(B, C, -1)
        contextual_embeddings = self.spatial_pooling(contextual_embeddings)
        contextual_embeddings = contextual_embeddings.view(B, 1, T, E).squeeze()

        # Project embedding dimension up to transformer's dimension
        contextual_embeddings = self.project_transformer(contextual_embeddings.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply transformer
        codex_embedding = self.codex_encoder(contextual_embeddings)

        # Project embeddings down to VQ dimension
        codex_embedding = self.pre_vq(codex_embedding.permute(0, 2, 1)).permute(0, 2, 1)

        # Quantize embeddings
        quantized, codes, commit_loss = self.quantizer(codex_embedding)

        # Project quantized embeddings up to transformer dimension
        z = self.post_vq(quantized.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply transformer
        z = self.codex_decoder(z)

        # Project embedding down to temporal encoding dimension
        z = self.project_temporal(z.permute(0, 2, 1)).permute(0, 2, 1)

        # Unpool in spatial dimension
        z = z.unsqueeze(1) # [B, 1, T, E]
        B, C, T, E = z.shape
        z = z.reshape(B, C, -1)
        z = self.spatial_unpooling(z)
        z = z.reshape(B, -1, T, E)

        with torch.no_grad():
            # Decode all channels individually in time with Encodec

            z = z.permute(1, 0, 2, 3) # [C, B, T, E]

            output_waves = []
            for batched_channel in z:
                batched_channel = batched_channel.permute(0, 2, 1) # [B, E, T]
                outputs = self.temporal_decoder(batched_channel) # [B, 1, T] ?
                outputs = outputs.squeeze(1)
                output_waves.append(outputs)
            
            output_waves = torch.stack(output_waves) # [C, B, T]
            output_waves = output_waves.permute(1, 0, 2) # [B, C, T]

            # Downsample signal to original sampling rate
            output_waves = F.interpolate(output_waves, size=original_samples)

        # Compute losses
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

    model = _make_attention_vqvae(
        shared_dim=266,
        temporal_dim=128, # Output dimension of encodec model
        transformer_dim=512, # Same as Encodec output (allows us to avoid additional work)
        sampling_rate=300,
        vq_dim=64,
        codebook_size=1024
    ).cuda()

    x = torch.randn(32, 266, 90).cuda()
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    model(x, dataset_id, subject_id)