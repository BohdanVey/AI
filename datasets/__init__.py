from . import datasets
from . import transforms
from . import aug
from . import sampler


def make_transforms(config):
    return transforms.Compose([
        getattr(transforms, transform_type)(**(config.params if config.params else {}))
        for transform_type, config in config.items()
    ])


def make_augmentations(config):
    augmentations = []
    if config == "None": return
    augs = aug.Augmentation(config.type)
    return augs


def make_dataset(config):
    dataset_params = config.params
    transforms_config = dataset_params.pop('transforms')
    augmentations_config = dataset_params.pop('augmentations')
    image_transforms = make_transforms(transforms_config.image) if transforms_config.image else None
    try:
        target_transforms = make_transforms(transforms_config.target) if transforms_config.target else None
    except:
        print("TEST DATASET")
        target_transforms = None
    augmentations = make_augmentations(augmentations_config) if augmentations_config else None

    dataset_init = getattr(datasets, config.type)
    dataset = dataset_init.from_config(
        dataset_params,
        image_transforms,
        target_transforms,
        augmentations
    )

    return dataset


def make_sampler(sampl, data):
    sampler_init = getattr(sampler, sampl)
    loader = sampler_init.from_config(
        data
    )

    return loader


def make_data_loader(config, dataset,sampl = None):
    data_loader_init = getattr(datasets, config.type)
    loader = data_loader_init.from_config(
        config.params,
        dataset,
        sampl
    )

    return loader
