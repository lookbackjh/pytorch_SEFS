import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Prediction(nn.Module):
    def __init__(self,in_features,out_features=1) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, out_features))
        self.layers.append(nn.Sigmoid())
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
        ## input feature size to hidden feature size      
        for i in range(num_layers - 1):
            ##create num layer with  hidden layer 
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(in_layer_activation())
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_features, out_features))
        ## translate 
        ## in_layer_activation of the last layer should be set differently depending on the situation (RElu for reconstruction, sigmoid for mask)
        self.layers.append(final_layer_activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SEFS_S_Phase(nn.Module):
    activation_table = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
    }
    def __init__(self, model_params):
        super(SEFS_S_Phase, self).__init__()
        ## i want pi to be a parameter with 0.5 as initial value with x_dim * 1 dimension
        self.pi=torch.nn.Parameter(torch.ones(model_params['x_dim'],1)*0.5)
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
        
        
        self.predictor = FCNet(self.z_dim, 1, self.num_layers_d, self.h_dim_d,
                               in_layer_activation=self.fc_activate_fn,
                               final_layer_activation=nn.Sigmoid)
        
        
        
        self.predictor_linear=Prediction(self.z_dim,1)
        ## wnat to use sigmod as final activation function
        
        ## for a multiclass classification task, final activation function must be softmax. 
        
        
    def get_pi(self):
        ##returns pi 
        return self.pi
    
    def estimate_probability(self, x_tilde):
        return self.predictor(x_tilde)
    
    def encode(self, x):
        return self.encoder(x)