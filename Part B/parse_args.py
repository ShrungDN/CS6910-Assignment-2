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
    parser.add_argument('-nfc', '--num_fc', type=int, default=1000, help='Number of neurons in the fully connected layer')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning Rate to be used')
    parser.add_argument('-da', '--data_aug', type=str, default='False', help='If "True", the data is augmented before training')
    parser.add_argument('-dr', '--dropout', type=float, default=0.3, help='Dropout parameter between 0 and 1')
    parser.add_argument('-a', '--activation', type=str, default='ReLU', help='Activation function to be used: "ReLU", "GELU", "SiLU" or "Mish"')
    
    parser.add_argument('-vp', '--view_preds', type=str, default='False', help='If "True", it will log image and its predictions on the wandb project (if -wl is also "True")')

    args = parser.parse_args()
    return args