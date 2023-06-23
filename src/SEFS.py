from pathlib import Path

import lightning as pl
import numpy as np
import torch.nn

from src.data.data_wrapper import DataWrapper
from src.supervision.trainer import STrainer
from src.self_supervision.trainer import SSTrainer
from lightning.pytorch.loggers import TensorBoardLogger


BASE_DIR = str(Path(__file__).resolve().parent.parent)


class SEFS:
    def __init__(self,
                 train_data: DataWrapper,  # input train_data should be a DataWrapper object
                 selection_prob,  # pre-selected selection probability. only used in self-supervision_phase phase
                 model_params,  # common for both phases
                 trainer_params,  # this is a dict containing all the parameters for both phases
                 ss_lightning_params,
                 s_lightning_params,
                 log_dir=BASE_DIR,  # default log directory
                 exp_name='',  # a string for indicating the name of the experiment
                 log_step=100,  # log every 100 steps
                 val_data = None,
                 ):

        if 'batch_size' not in ss_lightning_params:
            ss_lightning_params['batch_size'] = 32

        if 'batch_size' not in s_lightning_params:
            s_lightning_params['batch_size'] = 32

        ss_batch_size = ss_lightning_params.pop('batch_size')
        s_batch_size = s_lightning_params.pop('batch_size')

        self.log_dir = f"{log_dir}/logs"

        tb_logger_ss = TensorBoardLogger(
            save_dir=self.log_dir,
            name=exp_name,
            version="self_supervision_phase"
        )

        tb_logger_s = TensorBoardLogger(
            save_dir=self.log_dir,
            name=exp_name,
            version="supervision_phase"
        )

        self.train_ss_dataloader = train_data.get_self_supervision_dataloader(batch_size=ss_batch_size)
        self.train_s_dataloader = train_data.get_supervision_dataloader(batch_size=s_batch_size)

        if val_data is not None:
            self.val_ss_dataloader = val_data.get_self_supervision_dataloader(batch_size=ss_batch_size)
            self.val_s_dataloader = val_data.get_supervision_dataloader(batch_size=s_batch_size)

        else:
            self.val_ss_dataloader = None
            self.val_s_dataloader = None

        x_mean, x_dim, correlation_mat = train_data.get_data_info()

        self.self_supervision_phase = SSTrainer(
            x_mean=x_mean,
            correlation_mat=correlation_mat,
            selection_prob=selection_prob,
            model_params=model_params,
            trainer_params=trainer_params
        )

        self.self_supervision_phase_trainer = pl.Trainer(
            logger=tb_logger_ss,
            log_every_n_steps=log_step,
            default_root_dir=self.log_dir,
            **ss_lightning_params
        )

        self.supervision_phase = STrainer(
            x_mean=x_mean,
            correlation_mat=correlation_mat,
            selection_prob=selection_prob,
            model_params=model_params,
            trainer_params=trainer_params
        )

        self.supervision_trainer = pl.Trainer(
            logger=tb_logger_s,
            log_every_n_steps=log_step,
            default_root_dir=self.log_dir,
            **s_lightning_params
        )

    def train(self):
        self.self_supervision_phase_trainer.fit(self.self_supervision_phase,
                                                train_dataloaders=self.train_ss_dataloader,
                                                val_dataloaders=self.val_ss_dataloader
                                                )

        # load the trained weight of encoder from self-supervision_phase phase and assign to the supervision_phase phase
        trained_encoder = self.self_supervision_phase.model.encoder
        # note that the whole weights are saved under self.log_dir/checkpoints

        self.supervision_phase.model.encoder.load_state_dict(
            trained_encoder.state_dict()
        )

        self.supervision_trainer.fit(self.supervision_phase,
                                     train_dataloaders=self.train_s_dataloader,
                                     val_dataloaders=self.val_s_dataloader
                                     )



if __name__ == '__main__':
    from src.data.synthetic_data import SyntheticData

    data = DataWrapper(SyntheticData())

    sefs = SEFS(
        train_data=data,
        selection_prob=np.array([0.5 for _ in range(data.x_dim)]),
        model_params={
            'x_dim': data.x_dim,
            'z_dim': 10,
            'h_dim_e': 10,
            'num_layers_e': 2,

            'h_dim_d': 10,
            'num_layers_d': 2,

            'dropout': 0.1,
            'fc_activate_fn': torch.nn.ReLU
        },
        trainer_params={
            'alpha': 10,
            'beta': 0.1,
            'optimizer_params': {
                'lr': 1e-4,
            },
        },
        ss_lightning_params={
            'max_epochs': 10000,
            'precision': "16-mixed",
            'gradient_clip_val': 1.0,
            'batch_size': 256,
        },
        s_lightning_params={
            'max_epochs': 1000,
            'precision': "16-mixed",
            'gradient_clip_val': 1.0,
            'batch_size': 32,
        },
        exp_name='test',
        exp_version = 'beta-0.1'
    )

    sefs.train()



