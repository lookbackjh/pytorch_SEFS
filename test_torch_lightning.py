from src.self_supervision.trainer import SSTrainer
import numpy as np
import lightning as pl
import torch

def config(x_dim):
    model_params = {
        'x_dim': x_dim,
        'z_dim': 4,
        'h_dim_e': 10,
        'num_layers_e': 3,
        'h_dim_d': 10,
        'num_layers_d': 3,
        'fc_activate_fn': 'relu',
    }
    
    return model_params
    
def run_test():
    # create a random dataset 
    dataset = np.random.rand(100, 10).astype(np.float32)
    x_mean = np.mean(dataset, axis=0)
    x_dim = dataset.shape[1]
    correlation_mat = np.corrcoef(dataset, rowvar=False)
    
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
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    trainer = pl.Trainer(limit_train_batches=10, max_epochs=10)
    trainer.fit(model, dataloader)
    
    

    
if __name__ == '__main__':
    run_test()