import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Flatten, Softmax, Dropout
from torch.nn import Conv2d
from torch.nn import ReflectionPad2d, ZeroPad2d
from torch.nn import BatchNorm2d, LayerNorm
from torch.nn import Identity

class CNNModel(Module):
  def __init__(self, model_config):
    super(CNNModel, self).__init__()
    self.act = model_config['ACTIVATION']
    self.pool = model_config['POOL']
    self.dropout = Dropout(model_config['DROPOUT'])
    self.bn = model_config['BN']

    n = model_config['NUM_FILTERS']
    k = model_config['SIZE_FILTERS']

    self.conv1 = Sequential(Conv2d(3, n[0], kernel_size=k[0], padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv2 = Sequential(Conv2d(n[0], n[1], kernel_size=k[1], padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv3 = Sequential(Conv2d(n[1], n[2], kernel_size=k[2], padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv4 = Sequential(Conv2d(n[2], n[3], kernel_size=k[3], padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv5 = Sequential(Conv2d(n[3], n[4], kernel_size=k[4], padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    
    self.flatten = Flatten()

    self.fc1 = Sequential(Linear(1024, model_config['NFC']),
                          self.act())
    self.fc2 = Sequential(Linear(model_config['NFC'], 10),
                          Softmax(dim=1))
    
    if self.bn == 'True':
      self.conv1_bn = BatchNorm2d(n[0])
      self.conv2_bn = BatchNorm2d(n[1])
      self.conv3_bn = BatchNorm2d(n[2])
      self.conv4_bn = BatchNorm2d(n[3])
      self.conv5_bn = BatchNorm2d(n[4])
        
  def forward(self, x):
    if self.bn == 'True':
      x = self.conv1_bn(self.conv1(x))
      x = self.conv2_bn(self.conv2(x))
      x = self.conv3_bn(self.conv3(x))
      x = self.conv4_bn(self.conv4(x))
      x = self.conv5_bn(self.conv5(x))
    else:
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)

    x = self.flatten(x)
    x = self.dropout(x)
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x