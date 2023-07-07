import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# create OPLS cynthetic data class
class OPLS_synthetic:
    def __init__(self, label_size,unlabel_size, seed=12345) -> None:
        self.label_size=label_size
        self.unlabel_size=unlabel_size

    def generate_x1_9(self,y):
        size=y.shape[0]
        x1_9=[]
        #x1~x9 generation        
        for i in range(4):            
            xp=np.random.uniform(0,1,size)+0.8-2*y
            x1_9.append(xp)           
        for i in range(5):
            xp=np.random.uniform(0,1,size)-1.2-2*y
            x1_9.append(xp)
        return np.array(x1_9).T
    
    def generate_x10_30(self,y):
        size=y.shape[0]
        sigma = np.array([[12,10,8],[10,12,10],[8,10,23]])
        tot = []
        mu = np.array([1,2,3]).reshape(3,1)
        mu_new=np.multiply(mu,y.reshape(1,size))
        mu_new=mu_new.T
        for i in range(7):
                temp = []
                for j in range(size):
                    cur_mu=mu_new[j,:]
                    x = np.random.multivariate_normal(cur_mu,sigma,1)
                    temp.append(x)
                temp=np.array(temp).squeeze(1)       
                tot.append(temp)

        # np.array(tot) has shape (7,40,3) and i want to reshape it to (40,21)
        x10_30=np.concatenate(tot,axis=1)
        #x10_30=np.array(x10_30).T
        return x10_30
    
    def generate_composite(self, x):
        result_x=[]
        sample_size,generation_size=x.shape
        for i in range(generation_size):
            x31s=[]
            x32s=[]
            x33s=[]
            for j in range(sample_size):
                u1 = np.random.uniform(0,1,1)
                u2 = np.random.uniform(0,1,1)
                u3 = np.random.uniform(0,1,1)
                eps=np.random.normal(loc=0., scale=abs(x[j][i]/10), size=[1, 1])
                x31=u1*(x[j][i]+eps)/(u1+u2+u3)
                x32=u2*(x[j][i]+eps)/(u1+u2+u3)
                x33=u3*(x[j][i]+eps)/(u1+u2+u3)
                x31s.append(x31)
                x32s.append(x32)
                x33s.append(x33)
            result_x.append(x31s)
            result_x.append(x32s)
            result_x.append(x33s)
        return np.array(result_x).reshape(sample_size,generation_size*3)
    
    def create_data_part(self,size):
        y = np.random.binomial(1,0.4,size)
        x1_9=self.generate_x1_9(y)
        x10_30=self.generate_x10_30(y)
        x1_30=np.concatenate([x1_9,x10_30],axis=1)
        x31_120=self.generate_composite(x1_30)
        x121_390=self.generate_composite(x31_120)
        x391_1000 = np.random.normal(loc=0., scale=1.0, size=[size, 610]) # low correlation with y
        x=np.concatenate([x1_30,x31_120,x121_390,x391_1000],axis=1)
        return x,y
    
    def create_data(self):
        labeled_X,label_y=self.create_data_part(self.label_size)
        unlabeled_X,_=self.create_data_part(self.unlabel_size)

        scaler = MinMaxScaler()

        scaler.fit(np.concatenate([labeled_X, unlabeled_X], axis=0))

        labeled_X = scaler.transform(labeled_X)
        unlabeled_X = scaler.transform(unlabeled_X)
        return labeled_X,label_y,unlabeled_X