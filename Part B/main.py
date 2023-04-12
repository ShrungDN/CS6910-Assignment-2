from helper_functions import *
from torchsummary import summary

import numpy as np
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Flatten, Softmax, Dropout
from torch.nn import Conv2d
from torch.nn import ReflectionPad2d, ZeroPad2d
from torch.nn import BatchNorm2d, LayerNorm
from torch.nn import Identity
from torch.nn import ReLU, GELU, SiLU, Mish

def main():
    model, n_inp = get_model(name='vgg16')

    training_layer = Sequential(Linear(n_inp, 500), 
                                ReLU(), 
                                Dropout(0.3),
                                Linear(500, 10), 
                                Softmax(dim=1))