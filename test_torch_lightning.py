from src.self_supervision.trainer import SSTrainer
import numpy as np
import lightning as pl
import torch
from synthetic_data import create_data

def config(x_dim):
    model_params = {
        'x_dim': x_dim,
        'z_dim': 30,
        'h_dim_e': 100,
        'num_layers_e': 3,
        'h_dim_d': 100,
        'num_layers_d': 3,
        'fc_activate_fn': 'relu',
    }
    
    return model_params
    
def run_test():
    # create a random dataset 

    unlabeled_x, labeled_x,labeled_y = create_data()
    ## wnat unlabed_x to numpy float 32 bit
    unlabeled_x = unlabeled_x.astype(np.float32)




    #dataset = np.random.rand(100, 10).astype(np.float32)
    x_mean = np.mean(unlabeled_x, axis=0)
    x_dim = unlabeled_x.shape[1]
    correlation_mat = np.corrcoef(unlabeled_x, rowvar=False)
    
    # create a selection probability in shape (x_dim)
    pi_ = np.array([0.5 for _ in range(x_dim)])
    
    model_params = config(x_dim)
    
    trainer_params = {
        'alpha': 0.5,
        'optimizer_params' :{
            'encoder_lr': 1e-4,
            'decoder_x_lr': 1e-4,
            'decoder_m_lr': 1e-4,
        }
    }
    
    model = SSTrainer(
        x_mean=x_mean,
        correlation_mat=correlation_mat,
        selection_prob = pi_,
        model_params = model_params,
        trainer_params = trainer_params
    )
    
    dataloader = torch.utils.data.DataLoader(unlabeled_x, batch_size=32, shuffle=True)
    
    trainer = pl.Trainer(limit_train_batches=10, max_epochs=10)
    trainer.fit(model, dataloader)
    
    

    
if __name__ == '__main__':
    run_test()