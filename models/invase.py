import numpy as np
import torch.nn as nn
import torch
from .predictor import Predictor
from .selector import Selector
from .baseline import Baseline


class InvaseModel(nn.Module):
    def __init__(self, dim, label_dim, args):
        """
        PyTorch model for INVASE
        """
        super(InvaseModel, self).__init__()
        self.actor_h_dim = args.actor_h_dim
        self.critic_h_dim = args.critic_h_dim
        self.n_layer = args.n_layer
        self.activation = args.activation
        if self.activation == "relu":
            self.activation = nn.ReLU

        elif self.activation == "selu":
            self.activation = nn.SELU
        self.dim = dim
        self.label_dim = label_dim
        # Build the predictor
        self.critic = Predictor(self.dim, self.critic_h_dim, self.label_dim, self.n_layer, self.activation)
        # Build the selector
        self.actor = Selector(self.dim, self.actor_h_dim, self.n_layer, self.activation)
        self.baseline = Baseline(self.dim, self.critic_h_dim, self.label_dim, self.n_layer, self.activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp, fw_module="actor"):
        if fw_module == "selector":
            return self.actor(inp)
        elif fw_module == "predictor":
            return self.critic(inp)
        elif fw_module == "baseline":
            return self.baseline(inp)
        else:
            raise NotImplementedError("This was not supposed to be used.")

    def importance_score(self, x):
        feature_importance = self.actor(x)
        return feature_importance

    def predict(self, x):
        # Generate a batch of selection probability
        selection_probability = self.actor(x)
        # Sampling the features based on the selection_probability
        selection = torch.bernoulli(selection_probability)
        # Prediction
        y_hat = self.critic(x*selection)
        return y_hat
