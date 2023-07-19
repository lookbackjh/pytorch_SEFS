import argparse
import itertools
from datetime import datetime
import numpy as np
from src.data.synthetic_data import SyntheticData
from src.data.data_wrapper import DataWrapper
from src.SEFS import SEFS
from src.models_common import ACTIVATION_TABLE
import torch.multiprocessing as mp


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

    parser.add_argument("--embed_dim", type=int, default=16, help="dimension of embedding")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads in multi-head attention")
    parser.add_argument("--noise_std", type=float, default=1, help="standard deviation of noise")

    # trainer params
    parser.add_argument("--alpha", type=float, default=1, help="regularization coefficient for m in self-supervision phase")
    parser.add_argument("--beta", type=float, default=0.005, help="regularization coefficient for pi in supervision phase")
    parser.add_argument("--l1_coef", type=float, default=1e-5, help="regularization coefficient for l1 norm of weights")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--mask_type", type=str, default='hard', help="mask type: hard of bernoulli soft for relaxed bernoulli")
    parser.add_argument("--ent_coef", type=float, default=0.1, help="coefficient for entropy regularization of the feature mask in SS phase")

    # lightning params
    parser.add_argument("--ss_epochs", type=int, default=50000, help="training epochs for self-supervision phase")
    parser.add_argument("--s_epochs", type=int, default=25000, help="training epochs for supervision phase")

    parser.add_argument("--ss_batch_size", type=int, default=1024, help="batch size for self-supervision phase")
    parser.add_argument("--s_batch_size", type=int, default=32, help="batch size for supervision phase")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="gradient clip value in l2 norm")

    return parser.parse_args()


def is_debug():
    # check if debug mode is on
    import sys

    if sys.gettrace() is not None:
        return True

    else:
        return False


def get_log_dir(args):
    #
    # Do some jobs here with args to create a experiment name with the given arguments.
    # Deafult is set to return "test"
    # cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # exp_name = f'test_{cur_time}'

    debug = "debug" if is_debug() else ""

    exp_name = f"{debug}/attn_mask/ss-{args.ss_epochs}-s-{args.s_epochs}/beta-{args.beta}/" \
               f"l1_coef-{args.l1_coef}/mask-{args.mask_type}/noise-{args.noise_std}/embedding-mixed/ent_coef-{args.ent_coef}/v2"

    return exp_name


def dict_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def run(**param):
    args = parse_args()
    
    if is_debug():
        args.ss_epochs = 100
        args.s_epochs = 100

    data = DataWrapper(SyntheticData("twomoon"))
    val_data = DataWrapper(SyntheticData("twomoon", 456))
    
    for k, v in param.items():
        setattr(args, k, v)

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

        'embed_dim': args.embed_dim,
        'n_heads': args.n_heads,
        'noise_std': args.noise_std,
    }

    trainer_params = {
        'alpha': args.alpha,
        'beta': args.beta,
        'l1_coef': args.l1_coef,
        'optimizer_params': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        },
        'mask_type': args.mask_type,
        'ent_coef': args.ent_coef,
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
        train_data=data,
        val_data=val_data,
        selection_prob=np.array([0.5 for _ in range(data.x_dim)]),
        model_params=model_params,
        trainer_params=trainer_params,
        ss_lightning_params=ss_lightning_params,
        s_lightning_params=s_lightning_params,
        exp_name=get_log_dir(args),  # this is the name of the experiment.
        # you can change it to whatever you want using the function above.

        early_stopping_patience=10000
    )

    sefs.train()


def main():
    params = {
        'beta': [5],
        'ent_coef': [0.1, 0, -0.1],
        'mask_type': ['hard', 'soft'],
        'noise_std': [0, 1, 2],
    }

    if not is_debug():
        pool = mp.Pool(4)

        for param in dict_product(params):
            pool.apply_async(run, kwds=param)

        pool.close()
        pool.join()

    else:
        for param in dict_product(params):
            run(**param)



if __name__ == '__main__':
    main()
