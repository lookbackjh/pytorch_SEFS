import torch
import torch.nn as nn
import numpy as np
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


        # to initialize weights for the first layer ( process of getting z, ztilde)
        self.filter_weights=nn.Parameter(torch.ones(2*self.x_dim))
        self.filter_weights.requires_grad=True
        self.idx=np.arange(2*self.x_dim)
        self.feature_idx=self.idx[:self.x_dim]
        self.ko_inds=self.idx[self.x_dim:]


        mlp_layers = [nn.Linear(self.x_dim,self.z_dim) ]
        for i in range(self.num_layers_d-1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(self.z_dim,self.z_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(self.z_dim,1))
        self.mlp=nn.Sequential(*mlp_layers)


        self.predictor = FCNet(self.x_dim, 1, self.num_layers_d, self.h_dim_d,
                                in_layer_activation=self.fc_activate_fn,
                                final_layer_activation=None)
        # predictor : (x_dim, 1), sends the selected features to the predictor with the aim of predicting the probability of the feature being selected

        self.predictor_linear = nn.Sequential(
            nn.Linear(self.z_dim, 1),
        )
    

    def _fetch_filter_weight(self):
        return self.filter_weights
    
    def feature_importance(self, weight_scores=True):
        
        with torch.no_grad():
            layers=list(self.mlp.named_children())
            W=layers[0][1].weight.cpu().numpy().T
            for layer in layers[1:]:
                if isinstance(layer[1], nn.ReLU):
                    continue
                weight=layer[1].weight.cpu().numpy().T
                W=np.dot(W,weight)
            W=W.squeeze(-1)
            Z = self._fetch_filter_weight().cpu().numpy()
            feature_imp = Z[self.feature_idx] * W
            knockoff_imp = Z[self.ko_inds] * W
        
        return np.square(feature_imp-knockoff_imp)
        




