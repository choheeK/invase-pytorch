import torch.nn as nn


class Selector(nn.Module):
    def __init__(self, dim, actor_h_dim, n_layer, activation):
        super(Selector, self).__init__()
        layers = list()
        layers.append(nn.Linear(in_features=dim, out_features=actor_h_dim))
        layers.append(activation())
        for _ in range(n_layer - 2):
            layers.append(nn.Linear(actor_h_dim, actor_h_dim))
            layers.append(activation())

        layers.append(nn.Linear(actor_h_dim, dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        return self.model(inp)

