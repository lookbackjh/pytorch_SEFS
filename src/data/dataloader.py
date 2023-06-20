from torch.utils.data import Dataset
import torch
class CustomDataset(Dataset):
    def __init__(self,x,y):
        self.x_data =torch.from_numpy(x)
        self.y_data =torch.from_numpy(y)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = (self.y_data[idx])

        return x, y
