"""
This is a wrapper for the data module.
It is used to load the data and create the dataloaders and return the data info, such as x_mean, x_dim, correlation_mat

"""
import numpy as np
import pandas as pd
from typing import Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.synthetic_data import SyntheticData


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y


class DataWrapper:
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, SyntheticData], seed=12345):
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
        else:
            self.data = data

        self.x_mean, self.x_dim, self.correlation_mat = self.get_data_info()

    def get_data_info(self):
        if isinstance(self.data, SyntheticData):
            x_mean, x_dim, correlation_mat = self.data.get_data_info()
        else:
            x_mean, x_dim, correlation_mat = self._get_data_info(self.data)

        return x_mean, x_dim, correlation_mat

    def _get_data_info(self, data):
        x_mean = np.mean(data, axis=0)
        x_dim = data.shape[1]
        correlation_mat = np.corrcoef(data, rowvar=False)

        return x_mean, x_dim, correlation_mat

    def get_self_supervision_dataset(self):
        if isinstance(self.data, SyntheticData):
            self_supervision_data = self.data.get_self_supervised_dataset()

        else:
            # TODO: this will be used to get only the parts of the data that are needed for self-supervision_phase phase
            raise NotImplementedError("Non-synthetic data is not supported yet")
            pass

        return self_supervision_data

    def get_self_supervision_dataloader(self, batch_size):
        ss_dataset = self.get_self_supervision_dataset()
        ss_dataloader = DataLoader(ss_dataset, batch_size=batch_size, shuffle=True)
        return ss_dataloader

    def get_supervision_dataset(self):
        if isinstance(self.data, SyntheticData):
            self_supervision_data = self.data.get_supervised_dataset()

        else:
            # TODO: this will be used to get only the parts of the data that are needed for self-supervision_phase phase
            raise NotImplementedError("Non-synthetic data is not supported yet")
            pass

        return self_supervision_data

    def get_supervision_dataloader(self, batch_size):
        s_dataset = self.get_supervision_dataset()
        s_dataloader = DataLoader(
            CustomDataset(*s_dataset), batch_size=batch_size, shuffle=True
        )
        # supervision_phase dataset must be wrapped by CustomDataset class to be used in dataloader
        return s_dataloader
