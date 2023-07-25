    
from src.data.twomoon_synthetic import Twomoon_synthetic
from src.data.OPLS_synthetic import OPLS_synthetic
from src.data.Deep_Pink_synthetic   import Deep_Pink_synthetic
import numpy as np
class SyntheticData:

    def __init__(self, datatype, feat_num,label_size,unlabel_size,seed=12345) -> None:
        self.datatype=datatype
        self.feat_num=feat_num 
        self.label_size=label_size
        self.unlabel_size=unlabel_size

        if datatype=="twomoon":
            self.dataset = Twomoon_synthetic(1000,1000,seed=seed)
        
        elif datatype=="opls":
            self.dataset = OPLS_synthetic(40,1000,seed=seed)

        elif datatype=="deeppink":
            self.dataset = Deep_Pink_synthetic(40,1000,self.feat_num,seed=seed)

        else:
            raise ValueError("datatype should be twomoon, opls or deeppink")
        
        self.labeled_x, self.labeled_y, self.unlabeled_x = self.dataset.create_data()
            
    def get_self_supervised_dataset(self):
        return self.unlabeled_x.astype(np.float32)
    
    def get_important_feature_idx(self):
        return self.dataset.get_feat_importance() 

    def get_supervised_dataset(self):
        return self.labeled_x.astype(np.float32), self.labeled_y.astype(np.float32)

    def get_data_info(self):
        x_mean = np.mean(self.unlabeled_x, axis=0)
        x_dim = self.unlabeled_x.shape[1]
        correlation_mat = np.corrcoef(self.unlabeled_x, rowvar=False)
        return x_mean, x_dim, correlation_mat