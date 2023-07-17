    
from src.data.twomoon_synthetic import Twomoon_synthetic
from src.data.OPLS_synthetic import OPLS_synthetic
import numpy as np
import knockpy as kpy
class SyntheticData:

    def __init__(self, datatype="twomoon",seed=12345) -> None:
        self.datatype=datatype
        if datatype=="twomoon":
            self.labeled_x, self.labeled_y,self.unlabeled_x = Twomoon_synthetic(1000,1000,seed=seed).create_data(seed=seed)
        elif datatype=="opls":
            self.labeled_x, self.labeled_y,self.unlabeled_x = OPLS_synthetic(40,1000,seed=seed).create_data()       
            
    def get_self_supervised_dataset(self):
            return self.unlabeled_x.astype(np.float32)

    def get_supervised_dataset(self):
        return self.labeled_x.astype(np.float32), self.labeled_y.astype(np.float32)

    def get_data_info(self):
        x_mean = np.mean(self.unlabeled_x, axis=0)
        x_dim = self.unlabeled_x.shape[1]
        correlation_mat = np.corrcoef(self.unlabeled_x, rowvar=False)
        return x_mean, x_dim, correlation_mat
    
    def get_knockoff_supervised_dataset(self):
        k1=kpy.knockoffs.GaussianSampler(self.labeled_x)
        x_tilde=k1.sample_knockoffs().astype(np.float32)
        return np.concatenate([self.labeled_x,x_tilde],axis=1).astype(np.float32), self.labeled_y.astype(np.float32)


        
         