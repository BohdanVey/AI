import segmentation_models_pytorch as smp
from . import models


def make_model(config):
    try:
        model_init = getattr(smp, config.type)
        model = model_init(**config.params)
    except AttributeError:
        model_init = getattr(models, config.type)
        model = model_init.from_config(config)

    return model
