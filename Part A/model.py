import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Flatten, Softmax, Dropout
from torch.nn import Conv2d
from torch.nn import ReflectionPad2d, ZeroPad2d
from torch.nn import BatchNorm2d, LayerNorm
from torch.nn import Identity

class CNNModel(Module):
  def __init__(self, activation, pool, dropout, nfc, bn):
    super(CNNModel, self).__init__()
    self.act = activation
    self.pool = pool
    self.dropout = Dropout(dropout)
    if bn:
      self.conv1_bn = BatchNorm2d(64)
      self.conv2_bn = BatchNorm2d(64)
      self.conv3_bn = BatchNorm2d(64)
      self.conv4_bn = BatchNorm2d(64)
      self.conv5_bn = BatchNorm2d(64)
    else:
      self.conv1_bn = Identity()
      self.conv2_bn = Identity()
      self.conv3_bn = Identity()
      self.conv4_bn = Identity()
      self.conv5_bn = Identity()

    self.conv1 = Sequential(Conv2d(3, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2),
                            self.conv1_bn)
    self.conv2 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2),
                            self.conv2_bn)
    self.conv3 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2),
                            self.conv3_bn)
    self.conv4 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2),
                            self.conv4_bn)
    self.conv5 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2),
                            self.conv5_bn)
    
    self.flatten = Flatten()

    self.fc1 = Sequential(Linear(1024, nfc),
                          self.act())
    self.fc2 = Sequential(Linear(nfc, 10),
                          Softmax(dim=1))
    
    
        
  def forward(self, x):
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