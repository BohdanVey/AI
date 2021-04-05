import segmentation_models_pytorch as smp
from . import models
import torch


def make_model(config, loss_f=None):
    model_init = getattr(models, config.type)
    model = model_init.from_config(config)
    if loss_f:
        model.loss = loss_f
    try:
        model.load_state_dict(torch.load(config.load))
        print(f"Model loaded, path: {config.load}")
    except:
        print("Wrong Load")
    return model
