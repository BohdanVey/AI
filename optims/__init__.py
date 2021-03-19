from . import optims


def make_optimizer(config, model_parameters):
    optimizer_type = getattr(optims, config.type)
    optimizer = optimizer_type(
        model_parameters,
        **config.params
    )

    return optimizer
