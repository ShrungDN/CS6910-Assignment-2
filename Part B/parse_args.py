import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tdp', '--train_data_path', type=str, default='/content/inaturalist_12K/train', help='Path to directory with training data')
    parser.add_argument('-tedp', '--test_data_path', type=str, default='/content/inaturalist_12K/val', help='Path to directory with testing data')

    parser.add_argument('-wp', '--wandb_project', type=str, default='ME19B168_CS6910_Assignment2', help='Project name on WandB')
    parser.add_argument('-we', '--wandb_entity', type=str, default='ME19B168', help='Username on WandB')
    parser.add_argument('-wn', '--wandb_name', type=str, default='assgn2_logs', help='Display name of run on WandB')
    parser.add_argument('-wl', '--wandb_log', type=str, default='False', help='If "True", results are logged into WandB, specified by wandb_project and wandb_entity')
    
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-l', '--loss', type=str, default='CrossEntropyLoss', help='Loss function: "CrossEntropyLoss"')
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', help='Optimizer to be used: "Adam", "Adadelta", "Adagrad", "NAdam", or "RMSprop"')
    parser.add_argument('-dimsw', '--dimsw', type=int, default=256, help='Width of resized image')
    parser.add_argument('-dimsh', '--dimsh', type=int, default=256, help='height of resized image')
    parser.add_argument('-p', '--pool', type=str, default='MaxPool2d', help='Pooling Layer: "MaxPool2d", "AvgPool2d", "AdaptiveMaxPool2d" or "AdaptiveAvgPool2d"')
    parser.add_argument('-nfc', '--num_fc', type=int, default=1000, help='Number of neurons in the fully connected layer')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning Rate to be used')
    parser.add_argument('-da', '--data_aug', type=str, default='True', help='If "True", the data is augmented before training')
    parser.add_argument('-dr', '--dropout', type=float, default=0.0, help='Dropout parameter between 0 and 1')
    parser.add_argument('-a', '--activation', type=str, default='ReLU', help='Activation function to be used: "ReLU", "GELU", "SiLU" or "Mish"')
    parser.add_argument('-bn', '--batch_norm', type=str, default='True', help='Batch Norm will be used if set to "True"')
    parser.add_argument('-nf1', '--num_filters1', type=int, default=64, help='Number of filters in 1st convolution layer')
    parser.add_argument('-nf2', '--num_filters2', type=int, default=64, help='Number of filters in 2nd convolution layer')
    parser.add_argument('-nf3', '--num_filters3', type=int, default=64, help='Number of filters in 3rd convolution layer')
    parser.add_argument('-nf4', '--num_filters4', type=int, default=64, help='Number of filters in 4th convolution layer')
    parser.add_argument('-nf5', '--num_filters5', type=int, default=64, help='Number of filters in 5th convolution layer')
    parser.add_argument('-sf1', '--size_filters1', type=int, default=5, help='Size of filters in 1st convolution layer')
    parser.add_argument('-sf2', '--size_filters2', type=int, default=5, help='Size of filters in 2nd convolution layer')
    parser.add_argument('-sf3', '--size_filters3', type=int, default=5, help='Size of filters in 3rd convolution layer')
    parser.add_argument('-sf4', '--size_filters4', type=int, default=5, help='Size of filters in 4th convolution layer')
    parser.add_argument('-sf5', '--size_filters5', type=int, default=5, help='Size of filters in 5th convolution layer')
    
    parser.add_argument('-vp', '--view_preds', type=str, default='False', help='If "True", it will log image and its predictions on the wandb project (if -wl is also "True")')
    parser.add_argument('-vf', '--visualize_filters', type=str, default='False', help='If "True", it will log the visualization of filters in 1st convolutional layer')
    parser.add_argument('-sc', '--sweep_config', type=str, default='SC1', help='Used with wandb_train.py to choose which sweep config to use')
    args = parser.parse_args()
    return args