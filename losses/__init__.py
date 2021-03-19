import segmentation_models_pytorch as smp
from . import losses


def make_loss(config):
    # NOTE: segmentation_models_pytorch losses will be updated in new release
    loss_type = getattr(smp.utils.losses, config.type, None) or getattr(losses, config.type)
    loss = loss_type(
        **(config.params if config.params else {})
    )

    return loss
