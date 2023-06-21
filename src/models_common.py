import torch
import torch.nn as nn


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

        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(in_layer_activation())
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_features, out_features))

        if final_layer_activation is not None:
            self.layers.append(final_layer_activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
