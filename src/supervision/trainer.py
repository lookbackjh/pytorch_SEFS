from typing import Any, Optional

from .model import SEFS_S_Phase
import lightning as pl
import torch
import torch.nn.functional as F
import math


class STrainer(pl.LightningModule):
    def __init__(self,
                 x_mean, # column wise mean of the whole data
                 correlation_mat, # correlation matrix of the whole data
                 selection_prob,    # pre-selected selection probability
                 model_params,
                 trainer_params
                 ) -> None:
        super().__init__()
        
        self.model = SEFS_S_Phase(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = self.trainer_params['alpha']  # alpha coef for gate vec loss
        self.beta_coef=self.trainer_params['beta'] # beta coef for controlling the number of feature selected.  
        self.optimizer_params = trainer_params['optimizer_params']
        
        self.x_mean = self._check_input_type(x_mean) 
        self.R = self._check_input_type(correlation_mat)
        
        


        self.L = torch.linalg.cholesky(self.R+1e-6*torch.eye(self.R.shape[1])) # compute cholesky decomposition of correlation matrix beforehand
    
    def _check_input_type(self, x):
        # check if the input is a torch tensor, if not, convert it to torch tensor 
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(dtype=torch.float32)
        return x
    
    def _check_device(self, x):
        # check if the input is on the same device as the model
        if x.device != self.device:
            x = x.to(self.device)
        return x
    
    def Gaussian_CDF(self, x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2.)))
    
    def relaxed_multiBern(self, batch_size,x_dim, pi,tau):

        eps = torch.normal(mean=0., std=1., size=[x_dim, batch_size]).to(self.device)
        v = torch.matmul(self.L, eps)
        u = self.Gaussian_CDF(v)
        m=torch.sigmoid((torch.log(u)-torch.log(1-u)+torch.log(pi)-torch.log(1-pi))/tau).T

        return m
    
    def multi_bern(self, batch_size, x_dim):
        # self.pi: (x_dim)
        # correltaion_matrix: (x_dim, x_dim)
        
        # draw a standard normal random vector for self-supervision phase
        eps = torch.normal(mean=0., std=1., size=[x_dim, batch_size]).to(self.device)
        # shape: (x_dim, batch_size)
        
        # generate a multivariate Gaussian random vector
        v = torch.matmul(self.L, eps)
        # shape of self.L : (x_dim, batch_size)
        
        # aplly element-wise Gaussian CDF
        u = self.Gaussian_CDF(v)
        # shape of u: (x_dim, batch_size)
        
        # generate correlated binary gate, using a boolean mask
        m = (u < self.pi[:, None]).float()
        # shape of m: (x_dim, batch_size)
        
        return m.T
        
    def training_step(self, batch, batch_size):
        # torch.autograd.set_detect_anomaly(True)
        x,y=batch
        batch_size, x_dim = x.shape
        
        self.L = self._check_device(self.L)

        pi=self.model.get_pi()

        ## for relaxation pi must be in shape of (x_dim,1), thus we unsquueze the pi
        

        ## pi is also a trainable param. 
        self.x_mean = self._check_device(self.x_mean)
        
        # sample gate vector

        ## mask is a relaxed version for parameter pi to be trained. 

        m= self.relaxed_multiBern(batch_size, x_dim,pi,1.0)
        # shape of m: (batch_sizex, x_dim)
 
        ## if m is greater than 0.5 want to make it 1
        m=(m>0.5).float()
        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)
        ## want to change dtype of xtilde to float32    
        x_tilde=x_tilde.to(dtype=torch.float32)
        
        ## for encoder the parameters should bee transferred from self-supervision phase

        # get z from encoder 
        z = self.model.encoder(x_tilde)
        
        # estimate x_hat from decoder
        y_hat = self.model.predictor(z).squeeze(1)
        
        
        # compute loss
        # loss_y: loss for classification
        loss_y = F.binary_cross_entropy(y_hat,y)
        # total_loss by using coefficient beta, controls the number fo features selected.
        total_loss=loss_y+self.beta_coef*pi.sum()
        
   
        
        # logging losses
        self.log('loss/total', total_loss, prog_bar=True)
        self.log('loss/temp', loss_y, prog_bar=True)
        self.log('first pi', pi[0], prog_bar=True)

        return total_loss
    
    def configure_optimizers(self):
        # need 3 different optimizers for 3 different parts
        encoder_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['lr'])
        return [encoder_optimizer], []