import pandas as pd
import numpy as np
import torch 
import random
import pickle
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons
from .dataloader import CustomDataset

    
class SyntheticData:
    def __init__(self) -> None:
        self.unlabeled_x, self.labeled_x, self.labeled_y = self.create_data()
        pass
    def get_noisy_two_moons(self,n_samples=1000, n_feats=100, noise_twomoon=0.1, noise_nuisance=1.0, seed_=1234):
        X, Y = make_moons(n_samples=n_samples, noise=noise_twomoon, random_state=seed_)
        np.random.seed(seed_)
        N = np.random.normal(loc=0., scale=noise_nuisance, size=[n_samples, n_feats-2])
        X = np.concatenate([X, N], axis=1)
        Y_onehot = np.zeros([n_samples, 2])
        Y_onehot[Y == 0, 0] = 1
        Y_onehot[Y == 1, 1] = 1

        return X, Y, Y_onehot

    def get_blockcorr(self,X, block_size=10, noise_=0.5, seed_=1234): 
        '''
            noise 0.5 ~ 0.85 correlation
            noise 1.0 ~ 0.66 correlation
        '''
        for p in range(X.shape[1]):
            np.random.seed(seed_ + p)
            tmp   = X[:, [p]] + np.random.normal(loc=0., scale=noise_, size=[X.shape[0], block_size-1])

            if p == 0:
                X_new = np.concatenate([X[:, [p]], tmp], axis=1)
            else:
                X_new = np.concatenate([X_new, X[:, [p]], tmp], axis=1)    
        return X_new   

    def create_data(self):
        seed=12345
        sigma_n = 1.0
        max_labeled_samples=10
        blocksize=10
        tr_X, tr_Y, tr_Y_onehot = self.get_noisy_two_moons(n_samples=1000, n_feats=10, noise_twomoon=0.1, noise_nuisance=sigma_n, seed_=seed)
        UX, UY, UY_onehot       = self.get_noisy_two_moons(n_samples=1000, n_feats=10, noise_twomoon=0.1, noise_nuisance=sigma_n, seed_=seed+1)
        block_noise = 0.3
        tr_X = self.get_blockcorr(tr_X, blocksize, block_noise, seed)
        UX   = self.get_blockcorr(UX, blocksize, block_noise, seed+1)
        random.seed(seed)
        idx1 = random.sample(np.where(tr_Y==1)[0].tolist(), max_labeled_samples)
        idx0 = random.sample(np.where(tr_Y==0)[0].tolist(), max_labeled_samples)

        idx  = idx1 + idx0
        random.shuffle(idx)
        tr_X        = tr_X[idx]
        tr_Y        = tr_Y[idx]
        tr_Y_onehot = tr_Y_onehot[idx]
        return UX, tr_X,tr_Y

    def self_supervision_data(self):
        # returns dataloader for self-supervison task 
        unlabeled_x = self.unlabeled_x.astype(np.float32)
        dataloader = torch.utils.data.DataLoader(unlabeled_x, batch_size=32, shuffle=True)
        return dataloader 

    def supervision_data(self):
        ## returnss dataloader for supervision  task
        labeled_x = self.labeled_x.astype(np.float32)
        labeled_y = self.labeled_y.astype(np.float32)
        label_dataset = CustomDataset(labeled_x, labeled_y)
        dataloader = torch.utils.data.DataLoader(label_dataset, batch_size=20, shuffle=True)
        return dataloader
    
    def get_self_supervison_info(self):
        x_mean = np.mean(self.unlabeled_x, axis=0)
        x_dim = self.unlabeled_x.shape[1]
        correlation_mat = np.corrcoef(self.unlabeled_x, rowvar=False)
        return x_mean, x_dim, correlation_mat
    
    def get_supervision_info(self):
        x_mean = np.mean(self.labeled_x, axis=0)
        x_dim = self.labeled_x.shape[1]
        correlation_mat = np.corrcoef(self.labeled_x, rowvar=False)
        return x_mean, x_dim, correlation_mat





