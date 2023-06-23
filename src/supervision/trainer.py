from typing import Any, Optional

from src.supervision.model import SEFS_S_Phase
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
        self.save_hyperparameters()

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
        
        # draw a standard normal random vector for self-supervision_phase phase
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

    def validation_step(self, batch, batch_size):
        x, y = batch
        # what is the shape of x and y?
        batch_size, x_dim = x.shape

        self.L = self._check_device(self.L)

        pi = self.model.get_pi()
        pi.data.clamp_(0)
        
        self.x_mean = self._check_device(self.x_mean)

        # sample gate vector

        # create a relaxed multi-bernoulli distribution for generating a mask
        m = self.relaxed_multiBern(batch_size, x_dim, pi, 1.0)
        # shape of m: (batch_sizex, x_dim)

        # if m is greater than 0.5 want to make it 1
        # m = (m > 0.5).float()

        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)

        # get z from encoder
        z = self.model.encoder(x_tilde)

        # estimate x_hat from decoder
        y_hat_logit = self.model.predictor_linear(z).squeeze(1)

        # compute loss
        loss_y = F.binary_cross_entropy_with_logits(y_hat_logit, y, reduction='mean')  # loss for y_hat
        total_loss = loss_y + self.beta_coef * pi.sum(-1).mean()

        # logging losses
        self.log('supervision/loss/val_total', total_loss, prog_bar=True)
        self.log('supervision/loss/val_y', loss_y, prog_bar=True)

        # log histogram of pi tensor
        self.logger.experiment.add_histogram('supervision/val_pi', pi, self.current_epoch)
        
        for i in pi:
            self.log(f"supervision/metric/pi/{i}", pi[i], prog_bar=False)

        return total_loss

    def training_step(self, batch, batch_size):
        x, y = batch
        # what is the shape of x and y?
        batch_size, x_dim = x.shape
        
        self.L = self._check_device(self.L)

        pi = self.model.get_pi()

        self.x_mean = self._check_device(self.x_mean)
        
        # sample gate vector

        # create a relaxed multi-bernoulli distribution for generating a mask
        m = self.relaxed_multiBern(batch_size, x_dim, pi, 1.0)
        # shape of m: (batch_sizex, x_dim)
 
        # if m is greater than 0.5 want to make it 1
        #m = (m > 0.5).float()

        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)

        # get z from encoder
        z = self.model.encoder(x_tilde)
        
        # estimate x_hat from decoder
        y_hat_logit = self.model.predictor_linear(z).squeeze(1)

        # compute loss
        loss_y = F.binary_cross_entropy_with_logits(y_hat_logit, y,reduction='mean') # loss for y_hat
        total_loss=loss_y+self.beta_coef*pi.sum(-1).mean()

        # logging losses
        self.log('supervision/loss/train_total', total_loss, prog_bar=True)
        self.log('supervision/loss/train_y', loss_y, prog_bar=True)

        # log histogram of pi tensor
        self.logger.experiment.add_histogram('supervision/train_pi', pi, self.current_epoch)
        
        for i in pi:
            self.log(f"supervision/metric/pi/{i}", pi[i], prog_bar=False)

        return total_loss


    def configure_optimizers(self):
        # need 3 different optimizers for 3 different parts
        encoder_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['lr'])
        return [encoder_optimizer], []