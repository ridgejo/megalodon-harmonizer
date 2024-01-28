import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def _make_temp_spatial_vqvae(sampling_rate, vq_dim, codebook_size, shared_dim):

    model_name = "facebook/wav2vec2-base-960h"
    wav2vec2_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

    quantizer = VectorQuantize(
        dim=vq_dim,
        codebook_size=codebook_size,
        codebook_dim=16,
        use_cosine_sim=True,
        threshold_ema_dead_code=2,
        kmeans_init=True,
        kmeans_iters=10,
    )

    return TempSpatialVQVAE(
        w2v2=wav2vec2,
        w2v2_processor=wav2vec2_processor,
        quantizer=quantizer,
        sampling_rate=sampling_rate,
        shared_dim=shared_dim,
        vq_dim=vq_dim,
    )



class TempSpatialVQVAE(nn.Module):

    def __init__(self, w2v2, w2v2_processor, quantizer, sampling_rate, shared_dim, vq_dim):
        super(TempSpatialVQVAE, self).__init__()

        self.w2v2 = w2v2
        self.w2v2_processor = w2v2_processor
        self.sampling_rate = sampling_rate

        self.spatial_pooling = nn.Conv1d(
            in_channels=shared_dim,
            out_channels=1,
            kernel_size=1,
        )

        self.codex_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.pre_vq = nn.Conv1d(
            in_channels=768,
            out_channels=vq_dim,
            kernel_size=1,
        )

        self.quantizer = quantizer

        self.post_vq = nn.Conv1d(
            in_channels=vq_dim,
            out_channels=768,
            kernel_size=1,
        )

        self.codex_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.spatial_unpooling = nn.Conv1d(
            in_channels=1,
            out_channels=shared_dim,
            kernel_size=1,
        )

        self.embedding_pooling = nn.Conv1d(
            in_channels=768,
            out_channels=1,
            kernel_size=1,
        )


    def forward(self, x, dataset_id, subject_id):

        original_x = x.clone()

        with torch.no_grad():
            # Encode all channels individually in time with wav2vec2 embeddings

            # Upsample signal to wav2vec2 sampling rate
            x = F.upsample(x, scale_factor=16000 // self.sampling_rate)

            x = x.permute(1, 0, 2) # [C, B, T]

            original_size = x.shape[-1]

            contextual_embeddings = []
            for batched_channel in x:
                # warning: assumes zero mean and unit variance
                outputs = self.w2v2(batched_channel, output_hidden_states=True)["last_hidden_state"]
                contextual_embeddings.append(outputs)
            
            contextual_embeddings = torch.stack(contextual_embeddings) # [C, B, T, emb_dim]
            contextual_embeddings = contextual_embeddings.permute(1, 0, 2, 3) # [B, C, T, emb_dim]

        # Pool in spatial dimension and output [B, T, E]
        B, C, T, E = contextual_embeddings.shape
        contextual_embeddings = contextual_embeddings.view(B, C, -1)
        contextual_embeddings = self.spatial_pooling(contextual_embeddings)
        contextual_embeddings = contextual_embeddings.view(B, 1, T, E).squeeze()

        codex_embedding = self.codex_encoder(contextual_embeddings)

        # Quantize embeddings
        codex_embedding = self.pre_vq(codex_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        quantized, codes, commit_loss = self.quantizer(codex_embedding)

        # Signal is now at 49Hz
        # Need 6x upscale to get to 300Hz again

        # Decode signal
        z = self.post_vq(quantized.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.codex_decoder(z)

        # Spatial decoding
        z = z.unsqueeze(1) # [B, 1, T, E]
        B, C, T, E = z.shape
        z = z.view(B, C, -1)
        z = self.spatial_unpooling(z)
        z = z.view(B, -1, T, E)

        # Temporal decoding
        z = z.permute(1, 0, 3, 2) # [C, B, E, T]
        output_embeds = []
        for batched_channel in z:
            # TODO: Replace with simple convolutional decoder
            # [B, 768, 14] -> [B, 1, 90]
            B, E, T = batched_channel.shape
            # 49Hz -> 300Hz upscaling and single-channel embedding
            batched_channel = F.upsample(batched_channel, size=original_size)
            batched_channel = self.embedding_pooling(batched_channel).squeeze(1) # [B, E, 90] -> [B, 1, 90]
            output_embeds.append(batched_channel)
        
        output_embeds = torch.stack(output_embeds) # [C, B, T]
        output_embeds = output_embeds.permute(1, 0, 2) # [B, C, T]

        # Compute losses
        recon_loss = F.mse_loss(original_x, output_embeds)
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

        return output_embeds

if __name__ == "__main__":

    import torch

    model = _make_temp_spatial_vqvae(
        sampling_rate=300,
        vq_dim=64,
        codebook_size=1024
    ).cuda()

    x = torch.randn(32, 266, 90).cuda()
    dataset_id = "TestDataset"
    subject_id = "TestSubject"

    model(x, dataset_id, subject_id)