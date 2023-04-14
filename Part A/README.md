# CS6910
Shrung D N - ME19B168 - Assignment 2 - Part A

## Description of files:

**helper_functions.py**: 
Python file with helper functions - such as functions used to train the model, load the dataset, transform the dataset, etc.


**main.py**:
Python file that is used to train a single model. The hyperparameters used to train this model can be passed as arguments as shown later in this documentation.


**model.py**:
Python file with the structure of the CNN model used in the assignment. I have used a model with 5 convolution layers followed by 1 hidden dense layer and an output layer. 


**parse_args.py**:
Python file used to parse the arguments sent through the command line. It is used to interact with main.py and wandb_train.py via the command line.


**sweep_configs.py**:
Python file with various sweep configurations for hyperparameter tuning using the WandB API.


**sweeps.txt**:
Text file with the different sweep configurations run for hyperparameter tuning.


**wandb_train.py**:
Python file that makes use of main.py to iteratively train multiple models using different hyperparameters (as defined by the sweep configuration obtained from sweep_configs.py)


The sript to train a particular model is done through the main.py. The script used to tune hyperparameters is done through the wandb_train.py file.

## main.py Usage
```
usage: python3 main.py [-h --help] 
                       [-tdp --train_data_path] <string> Path to directory with training data 
                       [-tedp --test_data_path] <string> Path to directory with testing data
                       [-wp --wandb_project] <string> Name of WandB Project
                       [-we --wandb_entity] <string> Username of WandB user
                       [-wn --wandb_name] <string> Name of WandB run
                       [-wl --wandb_log] <"True", "False"> Uploads logs into WandB if True
                       [-e --epochs] <int> Number of epochs to train the model
                       [-b --batch_size] <int> Batch size for training
                       [-l --loss] <"CrossEntropyLoss"> Loss function to use for training
                       [-o --optimizer] <"Adam", "Adadelta", "Adagrad", "NAdam", "RMSprop"> Optimizer to use for training
                       [-dimsw --dimsw] <int> Width to resize input image to for training
                       [-dimsh --dimsh] <int> Height to resize input image to for training
                       [-p --pool] <"MaxPool2d", "AvgPool2d", "AdaptiveMaxPool2d", "AdaptiveAvgPool2d"> Pooling layer to use for training
                       [-nfc --num_fc] <int> Number of neurons in hidden dense layer
                       [-lr --learning_rate] <float> Learning rate to use for training
                       [-da --data_aug] <"True", "False"> Data is augmented during training if True
                       [-dr --dropout] <float> Dropout parameter to use  
                       [-a --activation] <"ReLU", "GELU", "SiLU", "Mish"> Activation function to use for training
                       [-bn --batch_norm] <"True", "False"> Batch norm used in model if True  
                       [-nf1 --num_filters1] <int> Number of filters to be used in 1st convolution layer
                       [-nf2 --num_filters2] <int> Number of filters to be used in 2nd convolution layer  
                       [-nf3 --num_filters3] <int> Number of filters to be used in 3rd convolution layer  
                       [-nf4 --num_filters4] <int> Number of filters to be used in 4th convolution layer  
                       [-nf5 --num_filters5] <int> Number of filters to be used in 5th convolution layer  
                       [-sf1 --size_filters1] <int> Size of filters to be used in 1st convolution layer  
                       [-sf2 --size_filters2] <int> Size of filters to be used in 2nd convolution layer  
                       [-sf3 --size_filters3] <int> Size of filters to be used in 3rd convolution layer  
                       [-sf4 --size_filters4] <int> Size of filters to be used in 4th convolution layer  
                       [-sf5 --size_filters5] <int> Size of filters to be used in 5th convolution layer  
                       [-vp --view_preds] <"True", "False> Logs image and predictions on WandB if True (-wl must also be True)
                       [-vf --visualize_filters] <"True", "False"> Logs filters of 1st convolution layer on WandB if True (-wl must also be True)     	
```


## wandb_train.py Usage
```
usage: python3 wandb_train.py [-h --help] 
                       [-tdp --train_data_path] <string> Path to directory with training data 
                       [-tedp --test_data_path] <string> Path to directory with testing data
                       [-wp --wandb_project] <string> Name of WandB Project
                       [-we --wandb_entity] <string> Username of WandB user
                       [-wn --wandb_name] <string> Name of WandB run
                       [-l --loss] <"cross_entropy", "mean_squared_error"> Loss function to use for training
                       [-dimsw --dimsw] <int> Width to resize input image to for training
                       [-dimsh --dimsh] <int> Height to resize input image to for training 
                       [-sc --sweep_config] <"SC1_1", "SC1_2", "SC1_3", "SC2", "SC3", "SC4_1", "SC4_2", "SC4_3", "SC4_4", "SC4_5", "SC4_6", "SC4_7", "SC4_8"> Sweep configuration to be used from sweep_configs.py for hyperparameter tuning 	
```


## Optimal Hyperparameters (along with other parameters) found through hyperparameter search:
```
epochs: 10
batch_size: 64
loss: CrossEntropyLoss
optimizer: Adam
dimsw: 256
dimsh: 256
pool: MaxPool2d
nfc: 1000
learning_rate: 0.0001
data_aug: False
dropout: 0.3
activation: GELU
batch_norm: True
nf1: 64
nf2: 64
nf3: 64
nf4: 64
nf5: 64
sf1: 3
sf2: 3
sf3: 3
sf4: 9
sf5: 9
```