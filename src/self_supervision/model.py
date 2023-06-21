import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models_common import FCNet


class SEFS_SS_Phase(nn.Module):
    activation_table = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
    }

    def __init__(self, model_params):
        super(SEFS_SS_Phase, self).__init__()

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

        self.encoder = FCNet(self.x_dim, self.z_dim, self.num_layers_e, self.h_dim_e,
                             in_layer_activation=self.fc_activate_fn)
        
        self.decoder_x = FCNet(self.z_dim, self.x_dim, self.num_layers_d, self.h_dim_d,
                                in_layer_activation=self.fc_activate_fn)
        
        self.decoder_m = FCNet(self.z_dim, self.x_dim, self.num_layers_d, self.h_dim_d,
                               in_layer_activation=self.fc_activate_fn,
                               final_layer_activation=None)
        # note that for generating a mask, we are supposed to use sigmoid activation
        # However, regarding with the numerical stability, we just output the logits and use BCE with logits
        
    def estimate_feature_vec(self, x_tilde):
        return self.decoder_x(x_tilde)
    
    def estimate_gate_vec(self, x_tilde):
        # this outputs the logits of the mask vector, not the probability
        return self.decoder_m(x_tilde)
    
    def encode(self, x):
        return self.encoder(x)