
# Reimplementaion of SEFS with pytorch 

## Introduction 
This repository is a reimplementation for paper [SEFS : Self-Supervision Enhanced Feature Selection with Correlated Gates(ICLR 2023)](https://openreview.net/pdf?id=oDFvtxzPOx)
with pytorch-lightning.  Original tensorflow code by author https://github.com/chl8856/SEFS



## Usage

install requirements.txt
Run `SEFS_test.py` to train and test the model. Note that self-supervision and supervision are conducted sequentially.

Since the model consists of both a self-supervision part and a supervision part, there are various hyperparameters to customize for each separate network. These include the number of epochs, layer number, and regularization parameter. You can experiment with different hyperparameters simply by adjusting the arguments in `parser.args`.

## Model
model is seperated in two parts, self supervision part and supervision part, you can see the model implementation detail in `src/self_supervision` and `src/supervision folder` ( check model, trainer to see how the implementation is done)


## Dataset
Here, we used the syntheteic dataset that is used in the paper (two moon synthetic dataset.)
You can add and implement your own datset after checking up the codes in `src/data` folder



## experiments
Code implemented on python 3.10~3.11 and cuda 11.8

As referemced in original paper, parameter $\beta$ is really important for fair result. 
Note that as pytorch used  mean instead of summation, $\beta=5$ in paper is equivalent to  $\beta=0.05$ in our implementation

Results can be automatically seen in tensorboard just by activating tensorboard --logdir logs after your implementation. 



