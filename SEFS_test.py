import argparse
from pathlib import Path

import lightning as pl
import numpy as np
import torch.nn
from src.data.synthetic_data import SyntheticData
from src.data.data_wrapper import DataWrapper
from src.SEFS import SEFS


def parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument("--z_dim", type=int, default=2**6, help="dimension of latent variable")

    parser.add_argument("--h_dim_e", type=int, default=100, help="dimension of hidden layers in encoder")
    parser.add_argument("--num_layers_e", type=int, default=3, help="number of hidden layers in encoder")

    parser.add_argument("--h_dim_d", type=int, default=10, help="dimension of hidden layers in decoder")
    parser.add_argument("--num_layers_d", type=int, default=3, help="number of hidden layers in decoder")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--fc_activate_fn", type=str, default="ReLU", help="activation function in fully connected layers")


    # trainer params
    parser.add_argument("--alpha", type=float, default=10, help="regularization coefficient for m in self-supervision phase")
    parser.add_argument("--beta", type=float, default=0.1, help="regularization coefficient for pi in supervision phase")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")

    # lightning params
    parser.add_argument("--ss_epochs", type=int, default=100000, help="trainin epochs for self-supervision phase")
    parser.add_argument("--s_epochs", type=int, default=10000, help="trainin epochs for supervision phase")

    parser.add_argument("--ss_batch_size", type=int, default=256, help="batch size for self-supervision phase")
    parser.add_argument("--s_batch_size", type=int, default=32, help="batch size for supervision phase")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="gradient clip value in l2 norm")

    return parser.parse_args()


def get_log_dir(args):
    #
    # Do some jobs here with args to create a experiment name with the given arguments.
    # Deafult is set to return "test"

    exp_name = 'test'

    return exp_name




def main():
    data = DataWrapper(SyntheticData())

    args = parse_args()

    model_params = {
            'x_dim': data.x_dim,
            'z_dim': args.z_dim,
            'h_dim_e': args.h_dim_e,
            'num_layers_e': args.num_layers_e,

            'h_dim_d': args.h_dim_d,
            'num_layers_d': args.num_layers_d,

            'dropout': args.dropout,
            'fc_activate_fn': args.fc_activate_fn,
    }

    trainer_params = {
            'alpha': args.alpha,
            'beta': args.beta,
            'optimizer_params': {
                'lr':  args.lr,
                'weight decay': args.weight_decay,
            },
        }

    ss_lightning_params = {
            'max_epochs': args.ss_epochs,
            'precision': "16-mixed",
            'gradient_clip_val': args.gradient_clip_val,
            'batch_size': args.ss_batch_size,
    }

    s_lightning_params = {
            'max_epochs': args.s_epochs,
            'precision': "16-mixed",
            'gradient_clip_val': args.gradient_clip_val,
            'batch_size': args.s_batch_size,
    }

    sefs = SEFS(
        data=data,
        selection_prob=np.array([0.5 for _ in range(data.x_dim)]),
        model_params=model_params,
        trainer_params=trainer_params,
        ss_lightning_params=ss_lightning_params,
        s_lightning_params=s_lightning_params,
        exp_name=get_log_dir(args),
    )

    sefs.train()


if __name__ == '__main__':
    main()
