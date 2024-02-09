

## Introduction 
This repository is a reimplementation for paper SEFS : Self-Supervision Enhanced Feature Selection with Correlated Gates(ICLR 2023)
with pytorch-lightning.  Original tensorboard code by author https://github.com/chl8856/SEFS



## Usage

run `SEFS_test.py` to train and test for the files
as the model is consisted with self-supervision part and supervision part, there are various  hypereparemeters to customize for each seperate network

We observed that the number of epochs in self-supervision and the coefficient beta are crucial for reproducing the results from the paper. 


## Dataset
here, we only used the syntheteic dataset that is used in the paper (two moon dataset.)
