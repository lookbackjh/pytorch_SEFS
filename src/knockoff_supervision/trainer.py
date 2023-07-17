import numpy as np
from matplotlib import pyplot as plt

from src.knockoff_supervision.model import KnockOff_S_Phase
import lightning as pl
import torch
import torch.nn.functional as F
import math

EPS = 1e-6

class KSTrainer(pl.LightningModule):
    def __init__(self,
                 x_mean, # column wise mean of the whole data
                 correlation_mat, # correlation matrix of the whole data
                 selection_prob,    # pre-selected selection probability
                 model_params,
                 trainer_params
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = KnockOff_S_Phase(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = self.trainer_params['alpha']  # alpha coef for gate vec loss
        self.beta_coef=self.trainer_params['beta'] # beta coef for controlling the number of feature selected.  
        self.l1_coef = self.trainer_params['l1_coef'] # l1 norm regul. coef
        self.optimizer_params = trainer_params['optimizer_params']
        
        self.x_mean = self._check_input_type(x_mean)  #x_mean: mean of the whole data, computed beforehand
        self.R = self._check_input_type(correlation_mat) # correlation matrix of the whole data ,computed beforehand

        self.L = torch.linalg.cholesky(self.R+1e-6*torch.eye(self.R.shape[1])) # compute cholesky decomposition of correlation matrix beforehand
        self.train_loss_y, self.train_loss_total = 0., 0.
        
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
    
    def first_layer(self, x):
        # first layer to get the weights of input vector sending to the filter nodes. 
        x=x[:,self.model.idx]
        x=self.model._fetch_filter_weight().unsqueeze(dim=0)*x
        x=x[:,self.model.feature_idx]-x[:,self.model.ko_inds]
        return x

    
    def __forward(self, batch, batch_size):
        x, y = batch
        # what is the shape of x and y?
        batch_size, x_dim = x.shape

        # want to concat x, x_knockoff


        x_knockoff=torch.rand(x.shape).to(self.device) # generate x_knockoff from uniform distribution
        x_combination = torch.concat([x, x_knockoff], dim=1) # x_tilde is the concat of x and x_knockoff

        # get z from encoder
        z=self.first_layer(x_combination) # z is the output of the first layer of the encoder



        # estimate x_hat from decoder
        y_hat_logit = self.model.mlp(z).squeeze(1)


        # compute loss
        loss_y = F.binary_cross_entropy_with_logits(y_hat_logit, y, reduction='mean')  # loss for y_hat


        return loss_y

    def validation_step(self, batch, batch_size):
        loss_y = self.__forward(batch, batch_size)
        
        # logging losses
        #self.log('supervision/val_total', total_loss, prog_bar=True, logger=False)
        self.log('supervision/val_y', loss_y, prog_bar=True, logger=False)
        
        return loss_y

    def training_step(self, batch, batch_size):
        loss_y = self.__forward(batch, batch_size)

        self.train_loss_y = loss_y.item()
        
        # logging losses
        imp=self.model.feature_importance()
        self.log('supervision/train_y', loss_y, prog_bar=True)
        self.log('supervision/feature_importance1',imp[0] , prog_bar=True)
        self.log('supervision/feature_importance3',imp[2] , prog_bar=True)
        # log histogram of pi tensor
        #self.logger.experiment.add_histogram('pi', self.model.get_pi(), self.current_epoch)

        return loss_y

    def configure_optimizers(self):
        # need 3 different optimizers for 3 different parts
        encoder_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['lr'], weight_decay=self.optimizer_params['weight_decay'])
        return [encoder_optimizer], []
    
    def _l1_weight_norm(self):
        l1_norm = 0.
        for param in self.model.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return l1_norm




