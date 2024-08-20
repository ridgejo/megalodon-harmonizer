import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super(FiLM, self).__init__()

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

        # f(x) = gamma and h(x) = beta where x is a conditional embedding
        # F => gamma * F + beta (feature-wise)

        # Function which predicts both gamma and beta (shared parameters)
        # self.film_projector = nn.Linear(in_features=self.embedding_dim, out_features=self.feature_dim * 2,)
        self.film_projector = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dim,
                out_features=self.feature_dim * 2,
            )
        )


    def forward(self, x, cond_embedding):
        """
        x : activations from the network [B, C, T]
        cond_embedding : conditional embedding
        """

        # Get Film conditioning factors
        # warning: Does not need batching as conditioning should be the same for the entire batch
        # out = nn.functional.linear(cond_embedding, self.film_projector.weight.clone(), self.film_projector.bias)
        out = self.film_projector(cond_embedding)
        gamma = out[:, : self.feature_dim]  # [B, C]
        beta = out[:, self.feature_dim :]  # [B, C]

        return x * gamma[:, :, None] + beta[:, :, None]


if __name__ == "__main__":
    import torch

    batch_size = 32
    embedding_dim = 4
    feature_dim = 16

    model = FiLM(
        embedding_dim=embedding_dim,
        feature_dim=feature_dim,
    )

    x = torch.rand(batch_size, feature_dim, 1024)  # [B, C, T]
    cond_embedding = torch.rand(embedding_dim).repeat(
        batch_size, 1
    )  # Same embedding for all elements of batch

    x_out = model(x, cond_embedding)

    breakpoint()
