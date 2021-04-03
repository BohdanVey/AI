from . import optims


def make_optimizer(config, model_parameters, i,step):
    optimizer_type = getattr(optims, config.type)
    try:
        now = int(i // step)

        lr = config.params.pop('lr')

        optimizer = optimizer_type(
            model_parameters,
            lr=lr[now],
            **config.params
        )
        config.params['lr'] = lr
        return optimizer

    except:  # We use cyclic learning rate
        base_lr = config.params.pop('base_lr')
        max_lr = config.params.pop('max_lr')
        step = config.params.pop('step')
        if i % (2 * step) > step:
            i = i % step
            lr = round(max_lr - (max_lr - base_lr) / step * i, 5)
        else:
            i = i % step
            lr = round(base_lr + (max_lr - base_lr) / step * i, 5)
        optimizer = optimizer_type(
            model_parameters,
            lr=lr,
            **config.params
        )
        config.params['base_lr'] = base_lr
        config.params['max_lr'] = max_lr
        config.params['step'] = step
        return optimizer
