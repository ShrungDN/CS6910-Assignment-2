import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta, Adagrad, Adam, NAdam, RMSprop
from torch.nn import ReLU, GELU, SiLU, Mish
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d

def get_transforms(data_aug, imgdims, mean, std):
  if data_aug:
    train_transform = transforms.Compose([
        transforms.Resize(imgdims),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.01, 1)),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
  else:
    train_transform = transforms.Compose([
        transforms.Resize(imgdims),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
  val_test_transform = transforms.Compose([
      transforms.Resize(imgdims),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
  ])
  return train_transform, val_test_transform

def get_data_loaders(train_data_path, train_transform, test_data_path, val_test_transform, batch_size):
    train_val_dataset = ImageFolder(root=train_data_path, transform=train_transform)
    test_dataset = ImageFolder(root=test_data_path, transform=val_test_transform)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, train_val_dataset.class_to_idx

def train(model, train_loader, optimizer, criterion, device):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_acc = 0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_acc += (preds == labels).sum().item() 
        
    loss = train_running_loss / counter
    acc = 100. * (train_running_acc / len(train_loader.dataset))
    return loss, acc

def validate(model, val_loader, criterion, device):
    print('Validation')
    model.eval()
    valid_running_loss = 0.0
    valid_running_acc = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)
            loss = criterion(outputs, labels)

            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_acc += (preds == labels).sum().item()

    loss = valid_running_loss / counter
    acc = 100. * (valid_running_acc / len(val_loader.dataset))
    return loss, acc

def eval_model(model, train_loader, val_loader, test_loader, criterion, device):
    model.eval()

    train_loss, train_acc = 0, 0
    val_loss, val_acc = 0, 0
    test_loss, test_acc = 0, 0

    counter = 0
    with torch.no_grad():
        for data in train_loader:
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_acc += (preds == labels).sum().item()
    train_loss = train_loss / counter
    train_acc = 100. * (train_acc / len(train_loader.dataset))

    counter = 0
    with torch.no_grad():
        for data in val_loader:
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_acc += (preds == labels).sum().item()
    val_loss = val_loss / counter
    val_acc = 100. * (val_acc / len(val_loader.dataset))

    counter = 0
    with torch.no_grad():
        for data in test_loader:
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            test_acc += (preds == labels).sum().item()
    test_loss = test_loss / counter
    test_acc = 100. * (test_acc / len(test_loader.dataset))

    model_metrics = {
        'train_loss' : train_loss,
        'train_acc' : train_acc,
        'val_loss' : val_loss,
        'val_acc' : val_acc,
        'test_loss' : test_loss,
        'test_acc' : test_acc
    }

    return model_metrics

from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta, Adagrad, Adam, NAdam, RMSprop

def get_optimizer(opt):
    if opt == 'Adam':
        return Adam
    elif opt == 'Adadelta':
        return Adadelta
    elif opt == 'Adagrad':
        return Adagrad
    elif opt == 'NAdam':
        return NAdam
    elif opt == 'RMSprop':
        return RMSprop
    else:
        raise Exception('Incorrect Optimizer')
    
def get_loss_func(loss_func):
    if loss_func == 'CrossEntropyLoss':
        return CrossEntropyLoss
    else:
        raise Exception('Incorrect Loss Function')

def get_activation(act):
    if act == 'ReLU':
        return ReLU
    elif act == 'GELU':
        return GELU
    elif act == 'SiLU':
        return SiLU
    elif act == 'Mish':
        return Mish
    else:
        raise Exception('Incorrect Activation Function')

def get_pooling(pool):
    if pool == 'MaxPool2d':
        return MaxPool2d
    elif pool == 'AvgPool2d':
        return AvgPool2d
    elif pool == 'AdaptiveMaxPool2d':
        return AdaptiveMaxPool2d
    elif pool == 'AdaptiveAvgPool2d':
        return AdaptiveAvgPool2d

def get_preds_plot(model, test_loader, class_to_idx, num_images=3):
    num_classes = len(class_to_idx.keys())
    idx_to_class = {class_to_idx[k]:k for k in class_to_idx.keys()}
    counter = {i:[] for i in range(num_classes)}
    c = num_images * num_classes
    while c > 0:
        for data in test_loader:
            if c <= 0:
                break
            images, labels = data
            images = images.numpy()
            labels = labels.numpy()
            for image, label in zip(images, labels):
                if len(counter[label]) < num_images:
                    counter[label].append(image)
                    c -= 1
    images = []
    labels = []
    for k in counter.keys():
        images = images + counter[k]
        labels = labels + [k]*len(counter[k])
    images = np.stack(images, axis=0)
    images = torch.Tensor(images)

    model.eval()
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    print(labels)
    print(preds)



    # fig, ax = plt.subplots(num_images, num_classes, figsize=(30, 15))
    # for i in range(num_classes):
    #     for j in range(num_images):
    #         ax[j, i].imshow(counter[i][j])
    # return fig

            # images = images.numpy()
            # labels = labels.numpy()

