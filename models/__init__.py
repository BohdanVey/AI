import segmentation_models_pytorch as smp
from . import models
import torch
import torch.nn as nn


def make_model(config, loss_f=None):
    model_init = getattr(models, config.type)
    model = model_init.from_config(config)
    if loss_f:
        model.loss = loss_f
    model_params = torch.load(config.load)
    new_params = {}
    for param_name in model_params:
        new_param_name = ".".join(param_name.split(".")[1:])
        new_params[new_param_name] = model_params[param_name]
    model.load_state_dict(new_params)
    print(f"Model loaded, path: {config.load}")
    print("Wrong Load")
    return model
