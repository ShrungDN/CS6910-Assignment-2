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
from torchvision import models

def main():
    model = models.inception_v3(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    # model, n_inp, img_dims = get_model(name='vgg16')

    training_layer = Sequential(Linear(2048, 500), 
                                ReLU(), 
                                Dropout(0.3),
                                Linear(500, 10), 
                                Softmax(dim=1))
    
    # model.classifier[-1] = training_layer
    
    model.fc = training_layer

    inceptionv3_input_dims = (3, 299, 299)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model.to(device, non_blocking=True)
    summary(model, inceptionv3_input_dims)
    print()

    