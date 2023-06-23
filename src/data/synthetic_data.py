import pandas as pd
import numpy as np
import torch 
import random
import pickle
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons

    
class SyntheticData:
    def __init__(self, seed=12345) -> None:
        self.unlabeled_x, self.labeled_x, self.labeled_y = self.create_data(seed=seed)

    def get_noisy_two_moons(self, n_samples=1000, n_feats=100, noise_twomoon=0.1, noise_nuisance=1.0, seed_=1234):
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

    def create_data(self, seed=12345, sigma_n=1.0, max_labeled_samples=10, blocksize=10, block_noise=0.3):
        labeld_X, labeled_y, _ = self.get_noisy_two_moons(n_samples=1000, n_feats=10, noise_twomoon=0.1, noise_nuisance=sigma_n, seed_=seed)
        unlabeled_X, unlabeled_y, _ = self.get_noisy_two_moons(n_samples=1000, n_feats=10, noise_twomoon=0.1, noise_nuisance=sigma_n, seed_=seed+1)

        labeld_X = self.get_blockcorr(labeld_X, blocksize, block_noise, seed)
        unlabeled_X = self.get_blockcorr(unlabeled_X, blocksize, block_noise, seed+1)

        # below is for creating a dataset with a few labeled samples
        rand_gen = np.random.default_rng(seed)

        true_label_idx = rand_gen.choice(np.where(labeled_y==1)[0].tolist(), max_labeled_samples)
        false_label_idx = rand_gen.choice(np.where(labeled_y==0)[0].tolist(), max_labeled_samples)
        
        total_labeled_idx = np.concatenate([true_label_idx, false_label_idx])
        rand_gen.shuffle(total_labeled_idx)

        labeld_X = labeld_X[total_labeled_idx]
        labeled_y = labeled_y[total_labeled_idx]

        scaler=MinMaxScaler()

        scaler.fit(np.concatenate([labeld_X, unlabeled_X], axis=0))

        labeld_X = scaler.transform(labeld_X)
        unlabeled_X = scaler.transform(unlabeled_X)

        return unlabeled_X, labeld_X, labeled_y

    def get_self_supervised_dataset(self):
        return self.unlabeled_x.astype(np.float32)

    def get_supervised_dataset(self):
        return self.labeled_x.astype(np.float32), self.labeled_y.astype(np.float32)

    def get_data_info(self):
        x_mean = np.mean(self.unlabeled_x, axis=0)
        x_dim = self.unlabeled_x.shape[1]
        correlation_mat = np.corrcoef(self.unlabeled_x, rowvar=False)
        return x_mean, x_dim, correlation_mat



