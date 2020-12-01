import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, dim, critic_h_dim, label_dim, n_layer, activation):
        super(Baseline, self).__init__()

        layers = list()
        layers.append(nn.Linear(in_features=dim, out_features=critic_h_dim))
        layers.append(activation())
        layers.append(nn.BatchNorm1d(critic_h_dim, eps=0.001, momentum=0.01))
        for _ in range(n_layer - 2):
            layers.append(nn.Linear(critic_h_dim, critic_h_dim))
            layers.append(activation())
            layers.append(nn.BatchNorm1d(critic_h_dim, eps=0.001, momentum=0.01))

        layers.append(nn.Linear(critic_h_dim, label_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        return self.model(inp)
