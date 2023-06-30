import argparse
from datetime import datetime
import numpy as np
from dateutil.parser import parser

from src.data.synthetic_data import SyntheticData
from src.data.data_wrapper import DataWrapper
from src.SEFS import SEFS
from src.models_common import ACTIVATION_TABLE


def parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument("--z_dim", type=int, default=10, help="dimension of latent variable")

    parser.add_argument("--h_dim_e", type=int, default=100, help="dimension of hidden layers in encoder")
    parser.add_argument("--num_layers_e", type=int, default=3, help="number of hidden layers in encoder")

    parser.add_argument("--h_dim_d", type=int, default=10, help="dimension of hidden layers in decoder")
    parser.add_argument("--num_layers_d", type=int, default=3, help="number of hidden layers in decoder")

    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--fc_activate_fn", type=str, default="relu", choices=list(ACTIVATION_TABLE.keys()),
                        help="activation function in fully connected layers")

    # trainer params
    parser.add_argument("--alpha", type=float, default=10, help="regularization coefficient for m in self-supervision phase")
    parser.add_argument("--beta", type=float, default=0.005, help="regularization coefficient for pi in supervision phase")
    parser.add_argument("--l1_coef", type=float, default=0.0001, help="regularization coefficient for l1 norm of weights")
    parser.add_argument("--tau", type=float, default=1.0, help="temperature parameter for gumbel softmax")
    parser.add_argument("--loss_weight", type=float, default=1.0, help="loss weight coefficient for ss loss and s loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")

    # lightning params
    parser.add_argument("--epochs", type=int, default=10000, help="trainin epochs for supervision phase")

    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for self-supervision phase")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="gradient clip value in l2 norm")

    return parser.parse_args()


def get_log_dir(args):
    #
    # Do some jobs here with args to create a experiment name with the given arguments.
    # Deafult is set to return "test"
    # cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # exp_name = f'test_{cur_time}'
    
    exp_name = f"beta-{args.beta}/no_pi_mean/l1_coef-{args.l1_coef}/logitpi"

    return exp_name


def main():
    for beta in [0.05]:
        _l1_coef = 1e-5
        data = DataWrapper(SyntheticData())
        # val_data = DataWrapper(SyntheticData(456))    # validation is currently not supported for semi-supervision
        val_data = None
        args = parse_args()
        
        args.l1_coef = _l1_coef
        args.beta = beta

        # NOTE: if you want to change the default values of the parameters, you can do it here.
        # i.e., args.z_dim = 2**7

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
                'l1_coef': args.l1_coef,
                'tau': args.tau,
                'loss_weight': args.loss_weight,

                'optimizer_params': {
                    'lr':  args.lr,
                    'weight_decay': args.weight_decay,
                },
            }

        lightning_params = {
                'max_epochs': args.epochs,
                'precision': "16-mixed",
                'gradient_clip_val': args.gradient_clip_val,
                'batch_size': args.batch_size,
        }

        sefs = SEFS(
            train_data=data,
            val_data=val_data,
            selection_prob=np.array([0.5 for _ in range(data.x_dim)]),
            model_params=model_params,
            trainer_params=trainer_params,
            lightning_params=lightning_params,
            exp_name=get_log_dir(args), # this is the name of the experiment.
                                        # you can change it to whatever you want using the function above.
                                        
            early_stopping_patience=1000
        )

        sefs.train()


if __name__ == '__main__':
    main()
