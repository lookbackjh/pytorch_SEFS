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
    
    def Gaussian_CDF(self, x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2.)))

    def generate_mask(self, x):
        tau = 1.0

        u = self.model.generate_mask(x).detach()
        # shape of u: (batch_size, x_dim)

        pi = self.model.get_pi()

        m = F.sigmoid(
            (torch.log(u)-torch.log(1-u)+torch.log(pi)-torch.log(1-pi))/tau)

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
        m = self.generate_mask(x)
        # shape of m: (batch_sizex, x_dim)

        # if m is greater than 0.5 want to make it 1
        # m = (m > 0.5).float()

        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)

        # get z from encoder
        z = self.model.encoder(x_tilde)

        # estimate x_hat from decoder
        y_hat_logit = self.model.predictor(z).squeeze(1)
        
        # compute loss
        loss_y = F.binary_cross_entropy_with_logits(y_hat_logit, y, reduction='mean')  # loss for y_hat

        pi_reg = pi.sum().mean()  # regularization term for pi
        
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

        # log histogram of pi tensor
        self.logger.experiment.add_histogram('pi', self.model.get_pi(), self.current_epoch)

        return total_loss

    def configure_optimizers(self):
        # need 3 different optimizers for 3 different parts
        encoder_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['lr'], weight_decay=self.optimizer_params['weight_decay'])
        return [encoder_optimizer], []
    
    def _l1_weight_norm(self):
        l1_norm = 0.
        for param in self.model.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return l1_norm

    def on_train_end(self) -> None:
        # save the image of pi to tensorboard
        pi_plot = self._plot_pi()
        self.logger.experiment.add_image('image/pi', pi_plot)

    def _plot_pi(self):
        # plot bar graph of pi and return the image as numpy array
        pi = self.model.get_pi()
        pi = pi.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))

        bars = ax.bar(np.arange(len(pi)), pi)

        bars[0].set_color('r')
        bars[10].set_color('r')

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
