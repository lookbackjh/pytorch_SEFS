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

"""
Create a class that is responsible for creating a sample of mask based on the interaction of the original input data.
One must learn the relationships of the data and sample a mask based on that.
The goal is to sample a mask in which highly dependent features are masked together.
We may use attention mechanism to learn the relationships between features.

"""




class MaskGenerator(nn.Module):
    """
    This class is responsible for creating a sample of mask based on the interaction of the original input data.
    One must learn the relationships of the data and sample a mask based on that.
    The goal is to sample a mask in which highly dependent features are masked together.
    We may use attention mechanism to learn the relationships between features.

    """
    def __init__(self, **model_params):
        super(MaskGenerator, self).__init__()

        self.embed_dim = model_params['embed_dim']
        self.n_heads = model_params['n_heads']
        self.noise_std = model_params['noise_std']

        self.attn_dim = self.embed_dim // self.n_heads

        self.act = nn.ReLU()

        mult = 2

        self.attention = nn.MultiheadAttention(self.embed_dim*mult, self.n_heads, dropout=0.1, bias=False, batch_first=True)
        # self.attention = nn.TransformerDecoderLayer(self.embed_dim, self.n_heads, dim_feedforward=self.embed_dim*2,
        #                                             activation=self.act, batch_first=True
        #                                             )

        self.linear_relation_embedder = nn.Linear(1, self.embed_dim)

        mult_fact = 2 if self.act.__class__.__name__ == 'SwiGLU' else 1

        self.non_linear_relation_embedder = nn.Sequential(
            nn.Linear(1, mult_fact * self.embed_dim),
            self.act,
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        self.attn_out_concat = nn.Linear(self.embed_dim*mult, 1, bias=False)



    def forward(self, x):
        # Caputre the relation of the data and mask some if they are noise
        # x: (batch_size, x_dim)
        batch_size = x.shape[0]
        x_dim = x.shape[1]

        device = x.device

        x = x.reshape(batch_size, -1, 1)

        linear_relation = self.linear_relation_embedder(x)
        non_linear_relation = self.non_linear_relation_embedder(x)

        mixed_relation = torch.cat([linear_relation,
                                    non_linear_relation
                                    ], dim=-1)

        # mixed_relation = non_linear_relation

        # mixed_relation = linear_relation + non_linear_relation
        # mixed_relation: (batch_size, n_heads, x_dim, attn_dim * 2)

        attn_output, attn_weight = self.attention(mixed_relation, mixed_relation, mixed_relation)
        # attn_output: (batch, x_dim, embed_dim), attn_weight: (batch, x_dim, x_dim)

        # attn_output = self.attention(mixed_relation, mixed_relation, mixed_relation)
        # attn_output: (batch, x_dim, embed_dim)

        attn_out_score = self.attn_out_concat(attn_output).reshape(batch_size, -1)
        # (batch, x_dim)

        # noise = torch.distributions.dirichlet.Dirichlet(
        #     torch.ones(x.shape[1])).sample().to(device)

        if self.noise_std > 0:
            noise = torch.normal(0, self.noise_std, (1, x_dim)).to(device)

        else:
            noise = 0

        final_input = attn_out_score + noise

        attn_out_score_clipped = torch.sigmoid(final_input).detach()

        attention_dist = F.softmax(final_input)

        return attn_out_score_clipped, attention_dist