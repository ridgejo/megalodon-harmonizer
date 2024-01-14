import torch.nn as nn

class SubjectBlock(nn.Module):
    """
    Differentiable subject-specific transformation block. Learned for each subject.
    Inspired by https://www.nature.com/articles/s42256-023-00714-5 who use a 1x1 convolution and no activation for their subject-specific layer.

    TODO: subject layer should be prefixed by a shared spatial attention and shared 1x1 convolution.
    """

    def __init__(self, subject_ids : list, in_channels, out_channels):
        """
        Args:
            subject_ids: A list of all subject IDs from all datasets
            in_channels:
            out_channels:
        """

        super(SubjectBlock, self).__init__()

        self.subject_encoders = {
            k : nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) for k in subject_ids
        }

        self.subject_decoders = {
            k : nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=1) for k in subject_ids
        }

    def forward(self, data, subject_id):
        return self.subject_encoders[subject_id](data)

    def decode(self, data, subject_id):
        return self.subject_decoders[subject_id](data)