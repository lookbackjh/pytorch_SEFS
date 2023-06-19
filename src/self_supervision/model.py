import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        ## input featuure 디멘션에서 hidden feature 디멘션으로 가는 linear layer        
        for i in range(num_layers - 1):
            ##num layer 개수만큼 hidden layer 생성
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(in_layer_activation())
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_features, out_features))
        ## 마지막 레이어의 in_layer_activation은 상황에 따라 다르게 설정해야함 (reconstruction 일 때는 RElu, mask 일 때는 sigmoid)
        self.layers.append(final_layer_activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
                               final_layer_activation=nn.Sigmoid)
        
        ## out put for mas should be in [0,1] so sigmoid function is used
        
    def estimate_feature_vec(self, x_tilde):
        return self.decoder_x(x_tilde)
    
    def estimate_gate_vec(self, x_tilde):
        return self.decoder_m(x_tilde)
    
    def encode(self, x):
        return self.encoder(x)