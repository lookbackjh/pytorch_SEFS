from typing import Any, Optional

from .model import SEFS_SS_Phase
import lightning as pl
import torch
import torch.nn.functional as F
import math


class SSTrainer(pl.LightningModule):
    def __init__(self,
                 x_mean, # column wise mean of the whole data
                 correlation_mat, # correlation matrix of the whole data
                 selection_prob,    # pre-selected selection probability
                 model_params,
                 trainer_params
                 ) -> None:
        super().__init__()
        
        self.model = SEFS_SS_Phase(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = self.trainer_params['alpha']  # alpha coef for gate vec loss
        self.optimizer_params = trainer_params['optimizer_params']
        
        self.x_mean = self._check_input_type(x_mean) 
        self.R = self._check_input_type(correlation_mat)
        
        self.L = torch.linalg.cholesky(self.R) # compute cholesky decomposition of correlation matrix beforehand
        self.pi = selection_prob if isinstance(selection_prob, torch.Tensor) else torch.from_numpy(selection_prob)
        
        self.automatic_optimization = False # we will use custom training loop
    
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
        
    def training_step(self, x, batch_size):
        batch_size, x_dim = x.shape
        
        self.L = self._check_device(self.L)
        self.pi = self._check_device(self.pi)
        self.x_mean = self._check_device(self.x_mean)
        
        encoder_opt, feature_est_opt, gate_vec_opt = self.optimizers()
        
        # sample gate vector
        m = self.multi_bern(batch_size, x_dim)
        # shape of m: (batch_sizex, x_dim)
        
        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)
        
        # get z from encoder
        z = self.model.encoder(x_tilde)
        
        # estimate x_hat from decoder
        x_hat = self.model.estimate_feature_vec(z)
        
        # estimate gate vector
        m_hat = self.model.estimate_gate_vec(z)
        
        # compute loss
        loss_x = F.mse_loss(x_hat, x)
        
        loss_m = self.alpha_coef * F.cross_entropy(m_hat, m)
        
        total_loss = loss_x + loss_m
        
        # Optimize encoder
        
        encoder_opt.zero_grad()
        self.manual_backward(total_loss, retain_graph=True)
        encoder_opt.step()
        
        # Optimize feature estimator
        feature_est_opt.zero_grad()
        self.manual_backward(loss_x, retain_graph=True)
        feature_est_opt.step()
        
        # optimize gate vector estimator
        gate_vec_opt.zero_grad()
        self.manual_backward(loss_m)
        gate_vec_opt.step()               
        
        # logging losses
        self.log('loss/x', loss_x)
        self.log('loss/m', loss_m)
        self.log('loss/total', total_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        # need 3 different optimizers for 3 different parts
        encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=self.optimizer_params['encoder_lr'])
        feature_est_optimizer = torch.optim.Adam(self.model.decoder_x.parameters(), lr=self.optimizer_params['decoder_x_lr'])
        gate_vec_optimizer = torch.optim.Adam(self.model.decoder_m.parameters(), lr=self.optimizer_params['decoder_m_lr'])
        
        return [encoder_optimizer, feature_est_optimizer, gate_vec_optimizer], []
        
        
        
        
        
        