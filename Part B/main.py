from helper_functions import *
from parse_args import *

import numpy as np
import torch
from torch.nn import  Sequential
from torch.nn import Linear, Softmax, Dropout
from torch.nn import Conv2d
from torch.nn import BatchNorm2d, LayerNorm
from torch.nn import ReLU, GELU, SiLU, Mish
from torchvision import models
from torchsummary import summary

def main(config, train_data_path, test_data_path, evaluate_model=False):
    inceptionv3_input_dims = (3, 299, 299)

    IMGDIMS = (inceptionv3_input_dims[1], inceptionv3_input_dims[2])
    MEAN, STD = config['MEAN_STD']
    DATA_AUG = config['DATA_AUG']
    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    EPOCHS = config['EPOCHS']
    OPTIM = get_optimizer(config['OPTIM'])
    LOSS_FUNC = get_loss_func(config['LOSS_FUNC'])
    ACTIVATION = get_activation(config['ACTIVATION'])
    NFC = config['NFC']
    DROPOUT = config['DROPOUT']
    
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

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model.to(device, non_blocking=True)
    summary(model, inceptionv3_input_dims)
    print()

    optimizer = OPTIM(model.parameters(), lr=LR)
    criterion = LOSS_FUNC()

    train_transform, val_test_transform = get_transforms(DATA_AUG, IMGDIMS, MEAN, STD)
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(train_data_path, train_transform, test_data_path, val_test_transform, BATCH_SIZE)

    logs = {
     'epochs': [],
     'train_loss': [],
     'train_acc': [],
     'val_loss': [],
     'val_acc': []
    }

    for epoch in range(EPOCHS):
        print(f"Training: Epoch {epoch+1} / {EPOCHS}")

    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
    print(f'Training: Loss = {train_epoch_loss:.4f} Accuracy = {train_epoch_acc:.4f}')

    val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, device)
    print(f'Validation: Loss = {val_epoch_loss:.4f} Accuracy = {val_epoch_acc:.4f}')

    logs['epochs'].append(epoch + 1)
    logs['train_loss'].append(train_epoch_loss)
    logs['train_acc'].append(train_epoch_acc)
    logs['val_loss'].append(val_epoch_loss)
    logs['val_acc'].append(val_epoch_acc)
    print('-'*50)

    if evaluate_model:
        model_metrics = eval_model(model, train_loader, val_loader, test_loader, criterion, device)
    else:
        model_metrics = None
    return model, logs, model_metrics, class_to_idx, test_loader

if __name__ == '__main__':
  args = parse_arguments()

  config = {'BATCH_SIZE': args.batch_size,
            'MEAN_STD': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            'DATA_AUG': args.data_aug,
            'LR': args.learning_rate,
            'EPOCHS': args.epochs,
            'OPTIM': args.optimizer,
            'LOSS_FUNC': args.loss,
            'DROPOUT': args.dropout,
            'ACTIVATION': args.activation,
            'NFC': args.num_fc,
            }
  
  model, logs, model_metrics, class_to_idx, test_loader = main(config, args.train_data_path, args.test_data_path, evaluate_model=True)

  print('Final Model Metrics:')
  print('Training: Accuracy = {} Loss = {}'.format(model_metrics['train_acc'], model_metrics['train_loss']))
  print('Validation: Accuracy = {} Loss = {}'.format(model_metrics['val_acc'], model_metrics['val_loss']))
  print('Testing: Accuracy = {} Loss = {}'.format(model_metrics['test_acc'], model_metrics['test_loss']))
  print()

  if args.wandb_log == 'True':
    import wandb
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    NAME = args.wandb_name

    wandb.login()
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    wandb.log({
          'BATCH_SIZE': config['BATCH_SIZE'],
          'DATA_AUG': config['DATA_AUG'],
          'LR': config['LR'],
          'EPOCHS': config['EPOCHS'],
          'OPTIM': config['OPTIM'],
          'LOSS_FUNC': config['LOSS_FUNC'],
          'DROPOUT': config['DROPOUT'],
          'ACTIVATION': config['ACTIVATION'],
          'NFC': config['NFC'],
    })

    for i in range(len(logs['epochs'])):
      wandb.log({
          'epochs': logs['epochs'][i],
          'train_acc': logs['train_acc'][i],
          'train_loss': logs['train_loss'][i], 
          'val_acc': logs['val_acc'][i], 
          'val_loss': logs['val_loss'][i]
      })

    wandb.log({'Train Accuracy': model_metrics['train_acc']})
    wandb.log({'Validation Accuracy': model_metrics['val_acc']})
    wandb.log({'Test Accuracy': model_metrics['test_acc']})
    
    if args.view_preds == 'True':
      preds_plot = get_preds_plot(model, test_loader, class_to_idx)
      wandb.log({'Predictions': wandb.Image(preds_plot)})
      preds_plot.savefig(f'ME19B168_{NAME}_preds_plot')
    
    wandb.finish()