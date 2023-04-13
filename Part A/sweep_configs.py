# Methodology of sweeps is done as in sweeps.txt

# Sweeps 1-8:
SC1_1 = {
    'name': 'SC1',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-3]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0, 0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True', 'False']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

SC1_2 = {
    'name': 'SC1',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0, 0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True', 'False']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

SC1_3 = {
    'name': 'SC1',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-5]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0, 0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True', 'False']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

# runs 13-13
SC2 = {
    'name': 'SC2',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['True']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

# Sweeps 14-16
SC3 = {
    'name': 'SC3',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU', 'SiLU', 'Mish']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

# Sweeps 17-24
SC4_1 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [9]},
        'sf2': {'values': [9]},
        'sf3': {'values': [3]},
        'sf4': {'values': [3]},
        'sf5': {'values': [3]}
    }
}

SC4_2 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [3]},
        'sf2': {'values': [3]},
        'sf3': {'values': [3]},
        'sf4': {'values': [9]},
        'sf5': {'values': [9]}
    }
}

SC4_3 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [8]},
        'nf2': {'values': [16]},
        'nf3': {'values': [32]},
        'nf4': {'values': [64]},
        'nf5': {'values': [128]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

SC4_4 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [8]},
        'nf2': {'values': [16]},
        'nf3': {'values': [32]},
        'nf4': {'values': [64]},
        'nf5': {'values': [128]},
        'sf1': {'values': [9]},
        'sf2': {'values': [9]},
        'sf3': {'values': [3]},
        'sf4': {'values': [3]},
        'sf5': {'values': [3]}
    }
}

SC4_5 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [8]},
        'nf2': {'values': [16]},
        'nf3': {'values': [32]},
        'nf4': {'values': [64]},
        'nf5': {'values': [128]},
        'sf1': {'values': [3]},
        'sf2': {'values': [3]},
        'sf3': {'values': [3]},
        'sf4': {'values': [9]},
        'sf5': {'values': [9]}
    }
}

SC4_6 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [128]},
        'nf2': {'values': [64]},
        'nf3': {'values': [32]},
        'nf4': {'values': [16]},
        'nf5': {'values': [8]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

SC4_7 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [128]},
        'nf2': {'values': [64]},
        'nf3': {'values': [32]},
        'nf4': {'values': [16]},
        'nf5': {'values': [8]},
        'sf1': {'values': [9]},
        'sf2': {'values': [9]},
        'sf3': {'values': [3]},
        'sf4': {'values': [3]},
        'sf5': {'values': [3]}
    }
}

SC4_8 = {
    'name': 'SC4',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['GELU']},
        'batch_norm': {'values': ['True']},
        'nf1': {'values': [128]},
        'nf2': {'values': [64]},
        'nf3': {'values': [32]},
        'nf4': {'values': [16]},
        'nf5': {'values': [8]},
        'sf1': {'values': [3]},
        'sf2': {'values': [3]},
        'sf3': {'values': [3]},
        'sf4': {'values': [9]},
        'sf5': {'values': [9]}
    }
}



def get_config(name):
    if name == 'SC1_1':
        return SC1_1
    elif name == 'SC1_2':
        return SC1_2
    elif name == 'SC1_3':
        return SC1_3
    elif name == 'SC2':
        return SC2
    elif name == 'SC3':
        return SC3
    elif name == 'SC4_1':
        return SC4_1
    elif name == 'SC4_2':
        return SC4_2
    elif name == 'SC4_3':
        return SC4_3
    elif name == 'SC4_4':
        return SC4_4
    elif name == 'SC4_5':
        return SC4_5
    elif name == 'SC4_6':
        return SC4_6
    elif name == 'SC4_7':
        return SC4_7
    elif name == 'SC4_8':
        return SC4_8