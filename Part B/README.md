# CS6910
Shrung D N - ME19B168 - Assignment 2 - Part B

## Description of files

**helper_functions.py**: 
Python file with helper functions - such as functions used to train the model, load the dataset, transform the dataset, etc.


**main.py**:
Python file that is used to train a single model from a pre-trained model. The hyperparameters used to train this model can be passed as arguments as shown later in this documentation.


**parse_args.py**:
Python file used to parse the arguments sent through the command line. It is used to interact with main.py via the command line.


The sript to train a model is done through the main.py.

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
                       [-nfc --num_fc] <int> Number of neurons in hidden dense layer
                       [-lr --learning_rate] <float> Learning rate to use for training
                       [-da --data_aug] <"True", "False"> Data is augmented during training if True
                       [-dr --dropout] <float> Dropout parameter to use  
                       [-a --activation] <"ReLU", "GELU", "SiLU", "Mish"> Activation function to use for training
                       [-vp --view_preds] <"True", "False> Logs image and predictions on WandB if True (-wl must also be True)	
```