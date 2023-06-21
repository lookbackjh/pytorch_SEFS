from src.self_supervision.trainer import SSTrainer
from src.supervision.trainer import STrainer
import numpy as np
import lightning as pl
import torch
from src.data.synthetic_data import SyntheticData


def config(x_dim):
    model_params = {
        'x_dim': x_dim,
        'z_dim': 10,
        'h_dim_e': 100,
        'num_layers_e': 3,
        'h_dim_d': 100,
        'num_layers_d': 3,
        'fc_activate_fn': 'tanh',
    }
    
    return model_params


def run_test_supervision():
    # create a random dataset
    data=SyntheticData()
    dataloader=data.supervision_dataloader()
    x_mean,x_dim,correlation_mat=data.get_data_info()
    # create a selection probability in shape (x_dim)d
    pi_ = np.array([0.5 for _ in range(x_dim)])
    model_params = config(x_dim)
    trainer_params = {
        'alpha': 10,
        'beta':1,
        'optimizer_params' :{
            'lr': 1e-4,
        }
    }
    
    model = STrainer(
        x_mean=x_mean,
        correlation_mat=correlation_mat,
        selection_prob = pi_,
        model_params = model_params,
        trainer_params = trainer_params
    )
    
    
    trainer = pl.Trainer(
        max_epochs=500000,
        #= clip_grad_value=1.0,
        )
    trainer.fit(model, dataloader)

def run_test_self_supervison():
    # create a random dataset 

    #dataset = np.random.rand(100, 10).astype(np.float32)
    data = SyntheticData()
    dataloader = data.self_supervision_dataloader()
    x_mean,x_dim,correlation_mat=data.get_data_info()
    
    # create a selection probability in shape (x_dim)
    pi_ = np.array([0.5 for _ in range(x_dim)])
    
    model_params = config(x_dim)
    
    trainer_params = {
        'alpha': 10,
        'beta':5,
        'optimizer_params' :{
            'lr': 1e-4,
        }
    }
    
    model = SSTrainer(
        x_mean=x_mean,
        correlation_mat=correlation_mat,
        selection_prob = pi_,
        model_params = model_params,
        trainer_params = trainer_params
    )
    
    trainer = pl.Trainer(
        max_epochs=500000,
        # clip_grad_value=1.0,
        )
    trainer.fit(model, dataloader)
    
    

    
if __name__ == '__main__':
    #run_test_self_supervison()
    run_test_supervision()
    #run_test_self_supervison()