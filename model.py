import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np
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

        self.x_hat=network_settings['x_hat'] ##computed beforhand
        self.pi_ = network_settings['pi_'] ## selected beforhand
        self.LT = network_settings['LT']     ##  computed beforehand
        self.batch_size = network_settings['batch_size']  ## selected beforhand
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
        
    def sample_gate_vector(self,x):
        # x: (batch_size, x_dim)
        # LT_matrix: (x_dim, x_dim) , Lower triangel of correlation matrix should be computed before the traing phase.(via choleskly decompostion) 
        # batch_size: batch_size
        # pi_: (x_dim, 1) , pi_ is a hyper parameter that controls the probability,
        ## given correlateion matrix, sample a binary vector from a multivariate Bernoulli distribution
        
        mask=self.mask_generation(self.pi_, self.LT, self.batch_size)
        x_tilde=mask*x+ (1-mask)*self.x_hat

        ## 애매한건 다 네트워크 입력값에 넣는걸로 해놨음. ex batchsize, x_hat(평균값), pi_ 등등


        return x_tilde,mask


    
    def mask_generation(self, pi_, L, batch_size):
        ## mb_size is a size of minibatch, pi_ as a hyper parameter that controls the probability, 
        epsilon = np.random.normal(loc=0., scale=1., size=[np.shape(L)[0], batch_size])
        g=np.matmul(L, epsilon)
        m = (1/2 * (1 + scipy.special.erf(g/np.sqrt(2)) ) < pi_).astype(float).T
        return m
        # generate a mask matrix
        
        
    def forward(self, x):
        # sample a binary vector from 
        x_tilde,mask=self.sample_gate_vector(x) ## xtilde and 원본 마스크에 대한 정보 저장
        z = self.encoder(x_tilde)
        x_hat = self.decoder_x(z) ## xtilde로 부터 복원된 x_hat
        m_hat = self.decoder_m(z) ## mask로부터 복원된 m_hat

        loss_recon = F.mse_loss(x_hat, x, reduction='none')
        loss_cross_entropy=F.binary_cross_entropy(m_hat, mask, reduction='none')

        return loss_recon,loss_cross_entropy, x_hat, m_hat