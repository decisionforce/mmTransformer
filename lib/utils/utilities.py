import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


def load_config_data(path: str) -> dict:
    """Load a config data from a given path
    :param path: the path as a string
    :return: the config as a dict
    """
    with open(path) as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_checkpoint(checkpoint_dir, model, optimizer, MR=1.0):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'BestMissRate': MR
    }

    torch.save(state, checkpoint_dir)
    print('model saved to %s' % checkpoint_dir)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)
    return model


def load_model_class(model_name):
    import importlib
    module_path = f'lib.models.TF_version.{model_name}'
    module_name = 'STF'
    target_module = importlib.import_module(module_path)
    target_class = getattr(target_module, module_name)
    return target_class


if __name__ == "__main__":
    
    state = torch.load('./models/demo.pt')
    state_dict = state['state_dict']

    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k,v in state_dict.items():
        
        components = k.split('.')

        if components[1] == 'STF':
            new_k: str = ['stacked_transformer',] + components[2:]
            new_k = '.'.join(new_k)
        else:
            new_k: str = components[1:]
            new_k = '.'.join(new_k)

        new_state_dict[new_k] = v
    
    new_state = {'state_dict':new_state_dict}
    torch.save(new_state, './models/new_demo.pt')




