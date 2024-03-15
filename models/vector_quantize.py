import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize as VQ


class VectorQuantize(nn.Module):
    def __init__(self, vq_config):
        super(VectorQuantize, self).__init__()
        self.vq = VQ(**vq_config)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, E, T] -> [B, T, E]
        quantized, indices, commit_loss = self.vq(x)
        quantized = quantized.permute(0, 2, 1)  # [B, T, E] -> [B, E, T]
        return quantized, indices, commit_loss
