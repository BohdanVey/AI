from . import datasets
from . import transforms
from . import augs


def make_transforms(config):
    return transforms.Compose([
        getattr(transforms, transform_type)(**(config.params if config.params else {}))
        for transform_type, config in config.items()
    ])


def make_augmentations(config):
    augmentations_init = getattr(augs, config.type)
    augmentations = augmentations_init(**config.params)

    return augmentations


def make_dataset(config):
    dataset_params = config.params
    transforms_config = dataset_params.pop('transforms')
    augmentations_config = dataset_params.pop('augmentations')

    image_transforms = make_transforms(transforms_config.image) if transforms_config.image else None
    target_transforms = make_transforms(transforms_config.target) if transforms_config.target else None
    augmentations = make_augmentations(augmentations_config) if augmentations_config else None

    dataset_init = getattr(datasets, config.type)
    dataset = dataset_init.from_config(
        dataset_params,
        image_transforms,
        target_transforms,
        augmentations
    )

    return dataset


def make_data_loader(config, dataset):
    data_loader_init = getattr(datasets, config.type)
    loader = data_loader_init.from_config(
        config.params,
        dataset
    )

    return loader
