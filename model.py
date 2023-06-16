import torch.nn as nn
class SEFS(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim, num_layers, dropout):
        super(SEFS, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc = nn.Linear(input_dim   , h_dim)
        ## want to add 3 layers for the fc
        ## writh thre fully connected layer at once 
        ## want activation functiopn between each layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            ## activation function
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.xhatdecoder = nn.Sequential(
            nn.Linear( z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            nn.ReLU(),
        )

        self.maskdecoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            nn.ReLU(),
        )
        ## fc with outputlayer size 
        self.fc_out = nn.Linear(h_dim, z_dim)
    
    def encode(self, x):
        x=self.encoder(x)
        x=self.fc_out(x)
        return x
    def xhatdecode(self,x):
        x=self.xhatdecoder(x)
        return x

    def maskdecode(self,x):
        x=self.maskdecoder(x)
        return x
    
    def forward(self, x): 
        x=self.encode(x)
        xtilde=self.xhatdecode(x)
        mask=self.maskdecode(x)
        return x,xtilde, mask