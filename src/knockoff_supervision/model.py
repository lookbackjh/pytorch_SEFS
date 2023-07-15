import torch
import torch.nn as nn

from src.models_common import FCNet, ACTIVATION_TABLE


class KnockOff_S_Phase(nn.Module):
    activation_table = ACTIVATION_TABLE

    def __init__(self, model_params):
        super(KnockOff_S_Phase, self).__init__()
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

        self.f_selector = nn.Linear(self.x_dim*2, self.x_dim, bias=False)
        # f_selecotor : (x_dim*2, x_dim), sends the concatenated features(x,x_knockoff) to the feature selector

        self.predictor = FCNet(self.x_dim, 1, self.num_layers_d, self.h_dim_d,
                                in_layer_activation=self.fc_activate_fn,
                                final_layer_activation=None)
        # predictor : (x_dim, 1), sends the selected features to the predictor with the aim of predicting the probability of the feature being selected

        self.predictor_linear = nn.Sequential(
            nn.Linear(self.z_dim, 1),
        )
    
    
    def f_select(self, x):
        return self.f_selector(x)