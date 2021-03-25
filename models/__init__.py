import segmentation_models_pytorch as smp
from . import models
import torch

def make_model(config):
    try:
        model_init = getattr(smp, config.type)
        model = model_init(**config.params)
    except AttributeError:
        model_init = getattr(models, config.type)
        model = model_init.from_config(config)
    try:
        model.load_state_dict(torch.load(config.load))
        print(f"Model loaded, path: {config.load    }")
    except:
        print("Wrong Load")
    return model
