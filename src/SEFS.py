from pathlib import Path

import lightning as pl
import numpy as np
import torch.nn

from src.data.data_wrapper import DataWrapper
from supervision.trainer import STrainer
from self_supervision.trainer import SSTrainer


BASE_DIR = str(Path(__file__).resolve().parent.parent)


class SEFS:
    def __init__(self,
                 data: DataWrapper, # input data should be a DataWrapper object
                 selection_prob,    # pre-selected selection probability. only used in self-supervision phase
                 model_params,  # common for both phases
                 trainer_params, # this is a dict containing all the parameters for both phases
                 ss_lightning_params,
                 s_lightning_params,
                 log_dir=BASE_DIR,  # default log directory
                 ):

        if 'batch_size' not in ss_lightning_params:
            ss_lightning_params['batch_size'] = 32

        if 'batch_size' not in s_lightning_params:
            s_lightning_params['batch_size'] = 32

        ss_batch_size = ss_lightning_params.pop('batch_size')
        s_batch_size = s_lightning_params.pop('batch_size')

        self.log_dir = log_dir

        self.ss_dataloader = data.get_self_supervision_dataloader(batch_size=ss_batch_size)
        self.s_dataloader = data.get_supervision_dataloader(batch_size=s_batch_size)
        x_mean, x_dim, correlation_mat = data.get_data_info()

        self.self_supervision_phase = SSTrainer(
            x_mean=x_mean,
            correlation_mat=correlation_mat,
            selection_prob=selection_prob,
            model_params=model_params,
            trainer_params=trainer_params
        )

        self.self_supervision_phase_trainer = pl.Trainer(
            default_root_dir=self.log_dir,
            **ss_lightning_params
        )

        self.supervision = STrainer(
            x_mean=x_mean,
            correlation_mat=correlation_mat,
            selection_prob=selection_prob,
            model_params=model_params,
            trainer_params=trainer_params
        )

        self.supervision_trainer = pl.Trainer(
            default_root_dir=self.log_dir,
            **s_lightning_params
        )

    def train(self):
        self.self_supervision_phase_trainer.fit(self.self_supervision_phase, self.ss_dataloader)

        # load the trained weight of encoder from self-supervision phase and assign to the supervision phase
        self.supervision.encoder = self.self_supervision_phase.encoder

        # save the trained encoder from self-supervision phase to the log dir
        self.self_supervision_phase.encoder.save_pretrained(f"{self.log_dir}/weights/weight.pt")

        self.supervision.encoder.load_state_dict(self.self_supervision_phase.encoder.state_dict())

        self.supervision_trainer.fit(self.supervision, self.s_dataloader)


if __name__ == '__main__':
    from src.data.synthetic_data import SyntheticData

    data = DataWrapper(SyntheticData())

    sefs = SEFS(
        data=data,
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
            'max_epochs': 100,
            'precision': "16-mixed",
            'gradient_clip_val': 1.0,
            'batch_size': 32,
        },
        s_lightning_params={
            'max_epochs': 100,
            'precision': "16-mixed",
            'gradient_clip_val': 1.0,
            'batch_size': 32,
        },
        log_dir=BASE_DIR + "/debug"
    )

    sefs.train()



