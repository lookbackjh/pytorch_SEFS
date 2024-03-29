import numpy as np
from matplotlib import pyplot as plt

from src.supervision.model import SEFS_S_Phase
import lightning as pl
import torch
import torch.nn.functional as F
import math

EPS = 1e-6


class STrainer(pl.LightningModule):
    def __init__(self,
                 x_mean, # column wise mean of the whole data
                 correlation_mat, # correlation matrix of the whole data
                 selection_prob,    # pre-selected selection probability
                 model_params,
                 trainer_params,
                 imp_feature_idx
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = SEFS_S_Phase(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = self.trainer_params['alpha']  # alpha coef for gate vec loss
        self.beta_coef=self.trainer_params['beta'] # beta coef for controlling the number of feature selected.  
        self.l1_coef = self.trainer_params['l1_coef'] # l1 norm regul. coef
        self.optimizer_params = trainer_params['optimizer_params']
        self.imp_feature_idx = imp_feature_idx
        self.x_mean = self._check_input_type(x_mean)  #x_mean: mean of the whole data, computed beforehand
        self.R = self._check_input_type(correlation_mat) # correlation matrix of the whole data ,computed beforehand

        self.L = self.R # compute cholesky decomposition of correlation matrix beforehand
        #self.L = torch.cholesky(self.R) # compute cholesky decomposition of correlation matrix beforehand we did not use this version because the dataaset is small and we can compute the cholesky decomposition of the correlation matrix beforehand.
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
    
    def Gaussian_CDF(self, x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2.)))
    
    def relaxed_multiBern(self, batch_size,x_dim, pi,tau):

        eps = torch.normal(mean=0., std=1., size=[x_dim, batch_size]).to(self.device)
        
        v = torch.matmul(self.L, eps).transpose(0,1)
        # v: (batch_size, x_dim)
        
        #generate a multivariate Gaussian random vector from gaussian copula
        u = self.Gaussian_CDF(v) + EPS

        #pi = pi.clamp(min=EPS, max=1-EPS)   # clamping pi for numerical stability dealing with log

        #relaxed multi-bernoulli distribution to make the gate vector differentiable
        m = F.sigmoid(
            (torch.log(u)-torch.log(1-u)+torch.log(pi)-torch.log(1-pi))/tau)

        return m
    
    def multi_bern(self, batch_size, x_dim):
        # self.pi: (x_dim)
        # correltaion_matrix: (x_dim, x_dim)
        
        # draw a standard normal random vector for self-supervision_phase phase
        eps = torch.normal(mean=0., std=1., size=[x_dim, batch_size]).to(self.device)
        # shape: (x_dim, batch_size)
        
        # generate a multivariate Gaussian random vector
        v = torch.matmul(self.L, eps).transpose(0,1)
        # shape of self.L : (batch_size, x_dim)
        
        # aplly element-wise Gaussian CDF
        u = self.Gaussian_CDF(v)
        # shape of u: (batch_size, x_dim)
        
        # generate correlated binary gate, using a boolean mask
        m = (u < self.pi).float()
        # shape of m: (batch_size, x_dim)
        
        return m

    def __forward(self, batch, batch_size):
        x, y = batch
        # what is the shape of x and y?
        batch_size, x_dim = x.shape

        self.L = self._check_device(self.L)

        pi = self.model.get_pi()
        ## clamp pi to be between 0 and 1

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
        y_hat_logit = self.model.predictor(z).squeeze(1)

        pi_reg = pi.mean()

        # compute loss
        loss_y = F.binary_cross_entropy_with_logits(y_hat_logit, y, reduction='mean')  # loss for y_hat

        beta = self.beta_coef
        total_loss = loss_y + beta * pi_reg

        return loss_y, total_loss

    def validation_step(self, batch, batch_size):
        loss_y, total_loss = self.__forward(batch, batch_size)
        
        # logging losses
        self.log('supervision/val_total', total_loss, prog_bar=True, logger=False)
        self.log('supervision/val_y', loss_y, prog_bar=True, logger=False)
        
        return total_loss

    def training_step(self, batch, batch_size):
        loss_y, total_loss = self.__forward(batch, batch_size)

        self.train_loss_y = loss_y.item()
        self.train_loss_total = total_loss.item()
        
        # logging losses
        self.log('supervision/train_total', total_loss, prog_bar=True)
        self.log('supervision/train_y', loss_y, prog_bar=True)
                # log pi0 and pi10
        pi= self.model.get_pi()

        self.log('supervision/pi0', pi[0,0], prog_bar=True)
        self.log('supervision/pi10', pi[0,10], prog_bar=True)

        # log histogram of pi tensor
        self.logger.experiment.add_histogram('pi', self.model.get_pi(), self.current_epoch)

        return total_loss

    def configure_optimizers(self):
        # need 3 different optimizers for 3 different parts
        encoder_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['lr'], weight_decay=self.optimizer_params['weight_decay'])
        return [encoder_optimizer], []
    
    def _l1_weight_norm(self):
        l1_norm = 0.

        for name, param in self.model.named_parameters():
            if name == 'pi':    # don't penalize pi
                continue

            if "mask_generator" in name:
                continue

            l1_norm += torch.sum(torch.abs(param))

        return l1_norm

    def on_train_end(self) -> None:
        # save the image of pi to tensorboard
        pi_plot = self._plot_pi()
        self.logger.experiment.add_image('image/pi', pi_plot)

    def _plot_pi(self):
        # plot bar graph of pi and return the image as numpy array
        pi = self.model.get_pi()
        pi = pi.detach().cpu().numpy().reshape(-1)

        fig, ax = plt.subplots(figsize=(10, 10))

        bars = ax.bar(np.arange(len(pi)), pi)

        if self.imp_feature_idx is not None:
            for i in self.imp_feature_idx:
                bars[i].set_color('r')

        ax.set_xlabel('feature index')
        ax.set_ylabel('pi')

        canvas = fig.canvas
        renderer = canvas.get_renderer()

        canvas.draw()

        buffer = renderer.buffer_rgba()

        pi_plot = np.asarray(buffer)[:, :, :3]
        # plt.show()    # uncomment this to show the plot
        out = pi_plot.transpose(2, 0, 1)
        return out
