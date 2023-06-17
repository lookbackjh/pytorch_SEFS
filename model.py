import torch
import torch.nn as nn
import torch.nn.functional as F


def Gaussian_CDF(x):
    return 0.5 * (1. + torch.erf(x / torch.sqrt(2.)))

class FCNet(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, hidden_features=100,
                 activation=nn.ReLU, dropout=0.0):
        super(FCNet, self).__init__()

        self.layers = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(in_features, hidden_features))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SEFS_SS_Phase(nn.Module):
    def __init__(self, input_dims, network_settings):
        super(SEFS_SS_Phase, self).__init__()

        self.x_dim = input_dims['x_dim']
        self.z_dim = input_dims['z_dim']
        
        self.reg_scale = network_settings['reg_scale']
        self.h_dim_e = network_settings['h_dim_e']
        self.num_layers_e = network_settings['num_layers_e']
        self.h_dim_d = network_settings['h_dim_d']
        self.num_layers_d = network_settings['num_layers_d']
        self.fc_activate_fn = network_settings['fc_activate_fn']

        self.encoder = FCNet(self.x_dim, self.z_dim, self.num_layers_e, self.h_dim_e,
                             self.fc_activate_fn)
        self.decoder_x = FCNet(self.z_dim, self.x_dim, self.num_layers_d, self.h_dim_d,
                                self.fc_activate_fn)
        self.decoder_m = FCNet(self.z_dim, self.x_dim, self.num_layers_d, self.h_dim_d,
                                self.fc_activate_fn)
        
    def sample_gate_vecotr(self, pi, correlation_matrix, num_samples):
        pass
        # correlation_matrix: (x_dim, x_dim) , correlation matrix should be computed before the traing phase. 
        # num_samples: batch_size
    
    def mask_generation():

        
        pass
        # generate a mask matrix
        
        
    def forward(self, x):
        # sample a binary vector from 
        
        z = self.encoder(x)
        x_hat = self.decoder_x(z)
        m_hat = self.decoder_m(z)

        return loss, x_hat, m_hat