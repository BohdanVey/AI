from . import optims


def make_optimizer(config, model_parameters, i):
    print(config)
    lr = config.params.pop('lr')

    optimizer_type = getattr(optims, config.type)
    optimizer = optimizer_type(
        model_parameters,
        lr=lr[i],
        **config.params
    )
    config.params['lr'] = lr
    return optimizer
