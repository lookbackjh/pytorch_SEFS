import torch
import torch.nn as nn

from src.models_common import FCNet, ACTIVATION_TABLE


class SemiSEFS(nn.Module):
    activation_table = ACTIVATION_TABLE

    def __init__(self, model_params):
        super(SemiSEFS, self).__init__()

        self.x_dim = model_params['x_dim']
        self.z_dim = model_params['z_dim']
        
        self.h_dim_e = model_params['h_dim_e']
        self.num_layers_e = model_params['num_layers_e']
        
        self.h_dim_d = model_params['h_dim_d']
        self.num_layers_d = model_params['num_layers_d']
        
        self.fc_activate_fn = model_params['fc_activate_fn']
        
        if isinstance(self.fc_activate_fn, str):
            if self.fc_activate_fn not in self.activation_table:
                raise ValueError(f'Invalid activation function name: {self.fc_activate_fn}')
            
            self.fc_activate_fn = self.activation_table[self.fc_activate_fn]

        self.pi = torch.nn.Parameter(torch.tensor([
            [0.0 for _ in range(self.x_dim)]
            ## logit for pi
        ])
        )

        self.encoder = FCNet(self.x_dim, self.z_dim, self.num_layers_e, self.h_dim_e,
                             in_layer_activation=self.fc_activate_fn)
        
        self.decoder_x = FCNet(self.z_dim, self.x_dim, self.num_layers_d, self.h_dim_d,
                                in_layer_activation=self.fc_activate_fn)
        
        self.decoder_m = FCNet(self.z_dim, self.x_dim, self.num_layers_d, self.h_dim_d,
                               in_layer_activation=self.fc_activate_fn,
                               final_layer_activation=None)

        self.predictor_linear = nn.Sequential(
            nn.Linear(self.z_dim, 1),
        )

        self.predictor = FCNet(self.z_dim, 1, self.num_layers_d, self.h_dim_d,
                        in_layer_activation=self.fc_activate_fn,
                        final_layer_activation=None)
        # note that for generating a mask, we are supposed to use sigmoid activation
        # However, regarding with the numerical stability, we just output the logits and use BCE with logits

    def get_pi(self):
        # returns pi
        return torch.sigmoid(self.pi)

    def estimate_probability(self, x_tilde):
        return self.predictor(x_tilde)

    def estimate_feature_vec(self, x_tilde):
        return self.decoder_x(x_tilde)
    
    def estimate_gate_vec(self, x_tilde):
        # this outputs the logits of the mask vector, not the probability
        return self.decoder_m(x_tilde)
    
    def encode(self, x):
        return self.encoder(x)