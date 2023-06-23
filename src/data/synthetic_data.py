import pandas as pd
import numpy as np
import torch 
import random
import pickle
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons

    
class SyntheticData:
    def __init__(self) -> None:
        self.unlabeled_x, self.labeled_x, self.labeled_y = self.create_data()

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
        blocksize = 10
        block_noise = 0.3

        tr_X, tr_Y, _ = self.get_noisy_two_moons(n_samples=1000, n_feats=10, noise_twomoon=0.1, noise_nuisance=sigma_n, seed_=seed)
        val_X, val_Y, _ = self.get_noisy_two_moons(n_samples=1000, n_feats=10, noise_twomoon=0.1, noise_nuisance=sigma_n, seed_=seed+1)

        tr_X = self.get_blockcorr(tr_X, blocksize, block_noise, seed)
        val_X = self.get_blockcorr(val_X, blocksize, block_noise, seed+1)

        random_gen = np.random.default_rng(seed)

        y_equals_1_idx = np.where(tr_Y == 1)[0].tolist()
        y_equals_0_idx = np.where(tr_Y == 0)[0].tolist()

        idx1 = random_gen.choice(y_equals_1_idx, max_labeled_samples)
        idx0 = random_gen.choice(y_equals_0_idx, max_labeled_samples)
        
        idx = idx1 + idx0
        random_gen.shuffle(idx)
        tr_X = tr_X[idx]
        tr_Y = tr_Y[idx]

        scaler=MinMaxScaler()
        scaler.fit(np.concatenate([tr_X, val_X], axis=0))
        tr_X    = scaler.transform(tr_X)
        val_X      = scaler.transform(val_X)
        return val_X, tr_X, tr_Y

    def get_self_supervised_dataset(self):
        return self.unlabeled_x.astype(np.float32)

    def get_supervised_dataset(self):
        return self.labeled_x.astype(np.float32), self.labeled_y.astype(np.float32)

    def get_data_info(self):
        x_mean = np.mean(self.unlabeled_x, axis=0)
        x_dim = self.unlabeled_x.shape[1]
        correlation_mat = np.corrcoef(self.unlabeled_x, rowvar=False)
        return x_mean, x_dim, correlation_mat



