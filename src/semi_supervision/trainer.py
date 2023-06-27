from typing import Any, Optional

from src.semi_supervision.model import SemiSEFS
import lightning as pl
import torch
import torch.nn.functional as F
import math


class SemiSEFSTrainer(pl.LightningModule):
    def __init__(self,
                 x_mean, # column wise mean of the whole data
                 correlation_mat, # correlation matrix of the whole data
                 selection_prob,    # pre-selected selection probability
                 model_params,
                 trainer_params
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SemiSEFS(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = self.trainer_params['alpha']  # alpha coef for gate vec loss
        self.l1_coef = self.trainer_params['l1_coef'] # l1 norm regul. coef
        self.tau = self.trainer_params['tau']   # temperature param for gumbel softmax in relaxed multi-bern
        self.loss_weight = self.trainer_params['loss_weight']
        self.optimizer_params = trainer_params['optimizer_params']
        
        self.x_mean = self._check_input_type(x_mean)  #x_mean: mean of the whole data, computed beforehand
        self.R = self._check_input_type(correlation_mat)# correlation matrix of the whole data ,computed beforehand
        
        self.L = torch.linalg.cholesky(self.R) # compute cholesky decomposition of correlation matrix beforehand

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
    
    def relaxed_multi_bern(self, batch_size, x_dim, pi):
        # self.pi: (x_dim)
        # correltaion_matrix: (x_dim, x_dim)
        
        # draw a standard normal random vector for self-supervision_phase phase
        eps = torch.normal(mean=0., std=1., size=[x_dim, batch_size]).to(self.device)
        # shape: (x_dim, batch_size)
        
        # generate a multivariate Gaussian random vector
        v = torch.matmul(self.L, eps)
        # shape of self.L : (x_dim, batch_size)
        
        # apply element-wise Gaussian CDF
        u = self.Gaussian_CDF(v)
        # shape of u: (x_dim, batch_size)
        
        # generate a relaxed mask
        _inner = pi.log() - (1-pi).log() + u.log() - (1-u).log()
        m = F.sigmoid((1/self.tau) * _inner)
        
        return m.T
        
    def cal_ss_loss(self, batch, batch_size, train=True):
        batch_size, x_dim = batch.shape

        pi = self.model.get_pi()
        # clamp pi to be between 0 and 1
        pi.data.clamp(0, 1)

        # sample gate vector
        m = self.relaxed_multi_bern(batch_size, x_dim, pi)
        # shape of m: (batch_sizex, x_dim)
        
        # generate feature subset
        # NOTE: This part is different to the previous version of SEFS. It uses a soft-mask in [0, 1]
        # NOTE: which is different from the hard mask
        x_tilde = torch.mul(m, batch) + torch.mul(1. - m, self.x_mean)
        
        # get z from encoder
        z = self.model.encoder(x_tilde)
        
        # estimate x_hat from decoder
        x_hat = self.model.estimate_feature_vec(z)
        
        # estimate gate vector
        m_hat = self.model.estimate_gate_vec(z)
        # shape of m_hat: (batch_size, x_dim) also it is the "logits" not the probability
        
        # compute loss
        loss_x = F.mse_loss(x_hat, batch)
        
        loss_m = self.alpha_coef * F.binary_cross_entropy_with_logits(m_hat, m)
        # loss_m = -(m*torch.log(m_hat) + (1-m)*torch.log(1-m_hat)).sum(-1).mean()
        # replace the binary cross entropy with the commented line if you want to observe a similar loss scale for the original code

        total_loss = loss_x + self.alpha_coef * loss_m

        # logging losses
        prefix = 'train' if train is True else 'val'

        self.log(f'{prefix}_x', loss_x, prog_bar=True)
        self.log(f'{prefix}_m', loss_m, prog_bar=True)
        
        return total_loss

    def cal_s_loss(self, batch, batch_size, train=True):
        x, y = batch
        # what is the shape of x and y?
        batch_size, x_dim = x.shape

        pi = self.model.get_pi()
        # clamp pi to be between 0 and 1
        pi.data.clamp(0, 1)

        # create a relaxed multi-bernoulli distribution for generating a mask
        m = self.relaxed_multi_bern(batch_size, x_dim, pi)
        # shape of m: (batch_sizex, x_dim)

        # generate feature subset
        x_tilde = torch.mul(m, x) + torch.mul(1. - m, self.x_mean)

        # get z from encoder
        z = self.model.encoder(x_tilde)

        # estimate x_hat from decoder
        y_hat_logit = self.model.predictor_linear(z).squeeze(1)

        # compute loss
        loss_y = F.binary_cross_entropy_with_logits(y_hat_logit, y, reduction='mean')  # loss for y_hat

        prefix = 'train' if train is True else 'val'

        # logging losses
        self.log(f'{prefix}_y', loss_y, prog_bar=True)

        return loss_y

    def training_step(self, batch, batch_size):
        self.L = self._check_device(self.L)
        self.x_mean = self._check_device(self.x_mean)

        # self-supervised loss input
        x_unlabeled = batch['x_unlabeled']
        x_labeled = batch['x_labeled']
        y = batch['y']

        ss_loss = self.cal_ss_loss(x_unlabeled, x_unlabeled.shape[0])
        s_loss = self.cal_s_loss((x_labeled, y), x_labeled.shape[0])

        l1_reg = self._l1_weight_norm()

        total_loss = ss_loss + self.loss_weight * s_loss + self.l1_coef * l1_reg

        self.log('total_loss', total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_size):
        # self-supervised loss input
        unlabeled_X = batch['unlabeled_X']
        labeled_X = batch['labeled_X']
        labeled_y = batch['labeled_y']      # just added labeled prefix for coherence

        ss_loss = self.cal_ss_loss(unlabeled_X, unlabeled_X.shape[0], train=False)
        s_loss = self.cal_s_loss((labeled_X, labeled_y), labeled_X.shape[0], train=False)

        total_loss = ss_loss + self.loss_weight * s_loss

        self.log(f'valid_total_loss', total_loss, prog_bar=True)

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