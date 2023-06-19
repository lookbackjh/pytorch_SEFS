import torch.nn as nn
import numpy as np
from src.model import SEFS_SS_Phase
from torchinfo import summary
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dims={'x_dim':100, 'z_dim':10}
network_settings={'batch_size':32, 'reg_scale':0.5, 'h_dim_e':100, 'num_layers_e':3, 'h_dim_d':100, 'num_layers_d':3, 'fc_activate_fn':nn.ReLU, 'x_hat':torch.randn(1,100), 'pi_':np.random.rand(100,1), 'LT':np.random.rand(100,100)}
model=SEFS_SS_Phase(input_dims, network_settings).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary(model, (32,100)) ## input size: sameplesize*feature size