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
        self.save_hyperparameters()
        
        self.model = SEFS_SS_Phase(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = trainer_params['alpha']  # alpha coef for gate vec loss
        self.l1_coef = trainer_params['l1_coef'] # l1 norm regul. coef
        self.optimizer_params = trainer_params['optimizer_params']
        self.mask_type = trainer_params['mask_type']
        self.ent_coef = trainer_params['ent_coef']

        
        self.x_mean = self._check_input_type(x_mean)  #x_mean: mean of the whole data, computed beforehand
        self.R = self._check_input_type(correlation_mat)# correlation matrix of the whole data ,computed beforehand
        
        self.L = torch.linalg.cholesky(self.R+1e-4*torch.eye(self.R.shape[1])) # compute cholesky decomposition of correlation matrix beforehand
        self.pi = selection_prob if isinstance(selection_prob, torch.Tensor) else torch.from_numpy(selection_prob).to(torch.float16) # selection probability for each feature, must be trained.
        self.pi.requires_grad = False

    
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
        u, attn_dist = self.model.generate_mask(x)
        # shape of u: (batch_size, x_dim)

        tau = 1.0
        pi = self.pi[None, :]

        relaxed_mask = F.sigmoid(
            (torch.log(u)-torch.log(1-u)+torch.log(pi)-torch.log(1-pi))/tau)

        return relaxed_mask, u, attn_dist

    def __forward(self, x, batch_size):
        batch_size, x_dim = x.shape

        self.L = self._check_device(self.L)
        self.pi = self._check_device(self.pi)
        self.x_mean = self._check_device(self.x_mean)

        # sample gate vector
        relaxed_mask, u, attn_dist = self.generate_mask(x)
        # shape of m: (batch_sizex, x_dim)

        if self.mask_type == 'hard':
            m = (u <= 0.5).to(torch.float32)

        elif self.mask_type == 'soft':
            m = relaxed_mask

        else:
            raise ValueError(f'Invalid mask type: {self.mask_type}')

        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)

        # get z from encoder
        z = self.model.encoder(x_tilde)

        # estimate x_hat from decoder
        x_hat = self.model.estimate_feature_vec(z)

        # estimate gate vector
        m_hat = self.model.estimate_gate_vec(z)
        # shape of m_hat: (batch_size, x_dim) also it is the "logits" not the probability

        # compute loss
        loss_x = F.mse_loss(x_hat, x)

        loss_m = self.alpha_coef * F.binary_cross_entropy_with_logits(m_hat, m)
        # loss_m = -(m*torch.log(m_hat) + (1-m)*torch.log(1-m_hat)).sum(-1).mean()
        # replace the binary cross entropy with the commented line if you want to observe a similar loss scale for the original code

        l1_norm = self._l1_weight_norm()

        attn_entropy = torch.distributions.Categorical(attn_dist).entropy().mean()

        total_loss = loss_x + self.alpha_coef * loss_m + self.l1_coef * l1_norm - self.ent_coef * attn_entropy
        # we want to maximize the entropy of the attention distribution, so we add a negative sign

        return loss_x, loss_m, l1_norm, total_loss, attn_entropy

    def validation_step(self, x, batch_size):
        loss_x, loss_m, l1_norm, total_loss, attn_entropy = self.__forward(x, batch_size)

        # log reconstruction loss
        self.log('self-supervision/val_x', loss_x, prog_bar=True)
        self.log('self-supervision/val_m', loss_m, prog_bar=True)
        self.log('self-supervision/val_total', total_loss, prog_bar=True)
        self.log('self-supervision/val_mask_entropy', attn_entropy, prog_bar=False)
        
    def training_step(self, x, batch_size):
        loss_x, loss_m, l1_norm, total_loss, attn_entropy = self.__forward(x, batch_size)
        
        # logging losses
        self.log('self-supervision/train_x', loss_x, prog_bar=True)
        self.log('self-supervision/train_m', loss_m, prog_bar=True)
        self.log('self-supervision/train_total', total_loss, prog_bar=True)
        self.log('self-supervision/train_mask_entropy', attn_entropy, prog_bar=False)
        # self.log('loss/temp', temp, prog_bar=True)

        # for name, param in self.model.mask_generator.named_parameters():
        #     if 'bias' in name:
        #         continue
        #
        #     self.logger.experiment.add_histogram(f'mask_generator/{name}', param, self.current_epoch)
        #
        #     if param.grad is not None:
        #         self.logger.experiment.add_histogram(f'mask_generator/{name}_grad', param.grad, self.current_epoch)

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