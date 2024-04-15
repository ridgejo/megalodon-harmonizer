import torch
import torch.nn as nn

class AttachSubject(nn.Module):
    """
    Attaches subject embedding to tensor of shape [B, E, T]
    """

    def __init__(self):
        super(AttachSubject, self).__init__()


    def forward(self, z, subject_embedding):

        subject_embedding = subject_embedding.unsqueeze(0).repeat(z.shape[0], 1) # [B, S]
        subject_embedding = subject_embedding.unsqueeze(-1).expand(-1, -1, z.shape[-1]) # [B, S, T]

        return torch.cat(
            (z, subject_embedding),
            dim=1
        )
