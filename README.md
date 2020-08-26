# PytorchTemplate
This repository contains my template for Pytorch projects. It contains code to train a simple CNN on MNIST data.


This template contains 4 python files:

train.py: this contains the training/validation code.
model.py: this contains the model (network architecture) code.
dataloader.py: this contains the dataloader template which will be dataset specific.
config.py: this contains the various configuration parameters which will be used in the other files.


To run, use:

python train.py

This should work with most torch versions, and can work on both CPU and GPU.

I tested this on a CPU using python 3.6.1, with torch version 1.5.0 and torchvision 0.6.0.
It is also tested on a GPU using python 3.5.2, with torch version 1.3.0 and torchvision 0.2.2.
