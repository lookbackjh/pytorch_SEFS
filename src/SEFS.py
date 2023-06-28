from pathlib import Path

import lightning as pl
import numpy as np
import torch
from matplotlib import pyplot as plt

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
                 early_stopping_patience=50,  # early stopping patience for both phases
                 ):
        
        torch.set_float32_matmul_precision('high')
        
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
            sub_dir ="self_supervision_phase"
        )

        tb_logger_s = TensorBoardLogger(
            save_dir=self.log_dir,
            name=exp_name,
            sub_dir="supervision_phase"
        )

        self.train_ss_dataloader = train_data.get_self_supervision_dataloader(batch_size=ss_batch_size)
        self.train_s_dataloader = train_data.get_supervision_dataloader(batch_size=s_batch_size)

        if val_data is not None:
            self.val_ss_dataloader = val_data.get_self_supervision_dataloader(batch_size=ss_batch_size, shuffle=False)
            self.val_s_dataloader = val_data.get_supervision_dataloader(batch_size=s_batch_size, shuffle=False)

        else:
            self.val_ss_dataloader = None
            self.val_s_dataloader = None

        x_mean, x_dim, correlation_mat = train_data.get_data_info()

        #################################################################################################
        # Self-Supervision Phase
        #################################################################################################

        ss_early_stopping = pl.pytorch.callbacks.EarlyStopping(
            monitor='self-supervision/val_total',
            patience=early_stopping_patience,
            mode='min',
            verbose=False
        )

        self.self_supervision_phase = SSTrainer(
            x_mean=x_mean,
            correlation_mat=correlation_mat,
            selection_prob=selection_prob,
            model_params=model_params,
            trainer_params=trainer_params
        )

        self.self_supervision_phase_trainer = pl.Trainer(
            logger=tb_logger_ss,
            check_val_every_n_epoch=10,
            default_root_dir=self.log_dir,
            callbacks=[ss_early_stopping],
            **ss_lightning_params
        )

        #################################################################################################
        # Supervision Phase
        #################################################################################################

        s_early_stopping = pl.pytorch.callbacks.EarlyStopping(
            monitor='supervision/val_total',
            patience=early_stopping_patience,
            mode='min',
            verbose=False,
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
            check_val_every_n_epoch=10,
            default_root_dir=self.log_dir,
            callbacks=[s_early_stopping],
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

        # save the image of pi to tensorboard
        pi_plot = self._plot_pi()
        self.supervision_trainer.logger.experiment.add_image('image/pi', pi_plot)

    def _plot_pi(self):
        # plot bar graph of pi and return the image as numpy array
        pi = self.supervision_phase.model.get_pi().squeeze(0)
        pi = pi.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.bar(np.arange(len(pi)), pi)

        ax.set_xlabel('feature index')
        ax.set_ylabel('pi')

        canvas = fig.canvas
        renderer = canvas.get_renderer()

        canvas.draw()

        buffer = renderer.buffer_rgba()

        pi_plot = np.asarray(buffer)[:, :, :3]
        # plt.show()    # uncomment this to show the plot
        out = pi_plot.transpose(2, 0, 1)
        return out
