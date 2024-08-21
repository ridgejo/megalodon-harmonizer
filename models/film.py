import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super(FiLM, self).__init__()

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

        # f(x) = gamma and h(x) = beta where x is a conditional embedding
        # F => gamma * F + beta (feature-wise)

        # Function which predicts both gamma and beta (shared parameters)
        self.film_projector = nn.Sequential()
        self.lin = nn.Linear(in_features=self.embedding_dim, out_features=self.feature_dim * 2,)
        # self.film_projector = nn.Sequential(
        #     nn.Linear(
        #         in_features=self.embedding_dim,
        #         out_features=self.feature_dim * 2,
        #     )
        # )
        self.film_projector.add_module("lin", self.lin)


    def forward(self, x, cond_embedding, stage="encode"):
        """
        x : activations from the network [B, C, T]
        cond_embedding : conditional embedding
        """

        # Get Film conditioning factors
        # warning: Does not need batching as conditioning should be the same for the entire batch
        if stage == "task":
            out = nn.functional.linear(cond_embedding, self.lin.weight.clone(), self.lin.bias)
        elif stage == "encode":
            print(f"Before projector cond_emb: {cond_embedding._version}", flush=True)
            out = self.film_projector(cond_embedding)
            print(f"After projector cond_emb: {cond_embedding._version}", flush=True)
        print(f"Before gamma out: {out._version}", flush=True)
        gamma = out[:, : self.feature_dim]  # [B, C]
        print(f"After gamma out: {out._version}", flush=True)
        print(f"Before beta out: {out._version}", flush=True)
        print(f"Before beta gamma: {gamma._version}", flush=True)
        beta = out[:, self.feature_dim :] # [B, C]
        print(f"After beta out: {out._version}", flush=True)
        print(f"After beta gamma: {gamma._version}", flush=True)

        print(f"Before film x: {x._version}", flush=True)
        film = x * gamma[:, :, None] + beta[:, :, None]
        print(f"After film x: {x._version}", flush=True)

        return film


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
