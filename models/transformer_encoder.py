import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, transformer_config):
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            **transformer_config["encoder_layer"]
        )
        out_channels = transformer_config["encoder_layer"]["d_model"]
        del transformer_config["encoder_layer"]

        self.project_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            **transformer_config,
        )

        self.project_out = nn.Conv1d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.project_in(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.project_out(x)
        return x