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


class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator, self).__init__()

        self.attn_dim = 32
        self.n_heads = 4
        attn_embedding_dim = self.attn_dim * self.n_heads
        non_linear_attn_embedding_dim = self.attn_dim // 2 * self.n_heads

        self.linear_relation_embedder = nn.Linear(1, attn_embedding_dim)
        self.non_linear_relation_embedder = nn.Sequential(
            nn.Linear(1, 2 * non_linear_attn_embedding_dim),
            SwiGLU(),
            nn.Linear(non_linear_attn_embedding_dim, attn_embedding_dim)
        )

        self.attn_out_concat = nn.Linear(attn_embedding_dim*2, 1)

    def forward(self, x):
        # Caputre the relation of the data and mask some if they are noise
        # x: (batch_size, x_dim)
        batch_size = x.shape[0]
        x_dim = x.shape[1]
        
        x = x.reshape(batch_size, -1, 1)

        linear_relation = self.linear_relation_embedder(x).reshape(batch_size, self.n_heads, -1, self.attn_dim)
        non_linear_relation = self.non_linear_relation_embedder(x).reshape(batch_size, self.n_heads, -1,
                                                                           self.attn_dim)

        mixed_relation = torch.cat([linear_relation, non_linear_relation], dim=-1)
        # mixed_relation: (batch_size, n_heads, x_dim, attn_dim * 2)

        attn_out = F.scaled_dot_product_attention(mixed_relation, mixed_relation, mixed_relation)
        # attn_out: (batch_size, n_heads, x_dim, attn_dim)

        attn_out_reshaped = attn_out.permute(0, 2, 1, 3).reshape(batch_size, x_dim, -1)
        # (batch_size, x_dim, attn_dim * n_heads)

        attn_out_score = self.attn_out_concat(attn_out_reshaped).reshape(batch_size, -1)
        # (batch_size, x_dim)

        attn_out_score_clipped = torch.sigmoid(attn_out_score)

        return attn_out_score_clipped