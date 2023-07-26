import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# create OPLS cynthetic data class
class Deep_Pink_synthetic:
    def __init__(self, label_size,unlabel_size,feat_dim, seed=12345) -> None:
        self.label_size=label_size
        self.unlabel_size=unlabel_size
        self.feat_dim=feat_dim
        self.imp_feat_idx=np.random.choice(self.feat_dim, int(self.feat_dim*0.3), replace=False)
    
    def g(self,x) :
        # a nonlinear function to make relation between x and y nonlinear. 
        return (x**3)/2
    
    def generate_xy(self,n):
        
        L=np.zeros((self.feat_dim,self.feat_dim))
        for j in range(self.feat_dim):
            for k in range(self.feat_dim):
                L[j,k]=0.5**(abs(j-k))
        temp = []
        for i in range(n):
            x = np.random.multivariate_normal([0]*self.feat_dim,L,1)
            temp.append(x)
        X_new=np.array(temp).squeeze(1)
        epsilon=np.random.normal(loc=0., scale=1., size=[n,1])
        beta=np.zeros((self.feat_dim,1))
        # i WANT TO SET 30 percent of beta as nonzero value coming from N(0,1.5)
        
        beta[self.imp_feat_idx, :] = np.random.normal(loc=0., scale=1.5, size=[int(self.feat_dim*0.3),1])
        Y=self.g(np.matmul(X_new, beta))+epsilon
        return X_new,Y
    
    def get_feat_importance(self):
        return self.imp_feat_idx

    def create_data(self):
        labeled_X,label_y=self.generate_xy(self.label_size)
        unlabeled_X,_=self.generate_xy(self.unlabel_size)
        
        scaler = MinMaxScaler()

        scaler.fit(np.concatenate([labeled_X, unlabeled_X], axis=0))
        
        scaler2=MinMaxScaler()
        scaler2.fit(label_y)
        label_y=scaler2.transform(label_y)
        
        labeled_X = scaler.transform(labeled_X)
        unlabeled_X = scaler.transform(unlabeled_X)

        return labeled_X,label_y,unlabeled_X