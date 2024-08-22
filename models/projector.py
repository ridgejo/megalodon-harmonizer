import torch.nn as nn


class Projector(nn.Module):
    """SSL projector (used in training, discarded during fine-tuning)"""

    def __init__(self, input_dim, hidden_dim):
        super(Projector, self).__init__()

        self.lin1 = nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            )
        self.act_fn = nn.ReLU()
        self.lin2 = nn.Linear(
                in_features=hidden_dim,
                out_features=input_dim,
            )

        # self.ssl_projector = nn.Sequential(
        #     nn.Linear(
        #         in_features=input_dim,
        #         out_features=hidden_dim,
        #     ),
        #     nn.ReLU(),
        #     nn.Linear(
        #         in_features=hidden_dim,
        #         out_features=input_dim,
        #     ),
        # )
        self.ssl_projector = nn.Sequential()
        self.ssl_projector.add_module("lin1", self.lin1)
        self.ssl_projector.add_module("act_fn", self.act_fn)
        self.ssl_projector.add_module("lin2", self.lin2)

    def forward(self, z, stage="task"):
        if stage == "encode":
            return self.ssl_projector(z)
        elif stage == "task":
            out = nn.functional.linear(z, self.lin1.weight.clone(), self.lin1.bias)
            out = self.act_fn(out)
            out = nn.functional.linear(out, self.lin2.weight.clone(), self.lin2.bias)
            return out



# y1 = torch.nn.functional.linear (x, self.lin1.weight.clone(), self.lin1.bias)

# vs

# y1 = self.lin1 (x)