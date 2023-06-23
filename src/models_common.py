import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


ACTIVATION_TABLE = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'swigluh': SwiGLU,  # SwiGLU is a custom activation function used in google PaLM's language model
    }


class FCNet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_layers=1,
                 hidden_features=100,
                 in_layer_activation=nn.ReLU,
                 final_layer_activation=nn.ReLU,
                 dropout=0.0):
        super(FCNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))

        mult_factor = 1

        if in_layer_activation().__class__.__name__ == 'SwiGLU':
            mult_factor = 2

        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features*mult_factor),
                    in_layer_activation(),
                    nn.Dropout(dropout),
                )
            )

        self.layers.append(nn.Linear(hidden_features, out_features))

        if final_layer_activation is not None:
            self.layers.append(final_layer_activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
