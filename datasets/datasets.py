import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import random
from .transforms import MixUp

from utils.image import read_tif


class AgroSegmentationDataset(Dataset):
    def __init__(self, csv_file, image_transforms, target_transforms, augmentations, labels=None):
        """
        Args:
            csv_file (string): Path to the csv file with data locations.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.image_transforms = image_transforms
        self.mask_transforms = target_transforms
        self.augmentations = augmentations

        self.__labels = np.load(labels)

    @classmethod
    def from_config(cls, config, image_transforms, target_transforms, augmentations):
        return cls(
            csv_file=config.csv_path,
            image_transforms=image_transforms,
            target_transforms=target_transforms,
            augmentations=augmentations,
            labels=config.labels_path
        )



class AgroVision2021DatasetNew(AgroSegmentationDataset):
    """AgroVision Dataset"""

    def __init__(self, csv_file, image_transforms=None, target_transforms=None, augmentations=None,
                 labels=None):
        self.csv_file = pd.read_csv(csv_file)
        self.image_transforms = image_transforms
        self.mask_transforms = target_transforms
        self.augmentations = augmentations
        self.__labels = np.load(labels)

    def __getitem__(self, item):
        row = self.csv_file.iloc[item]
        pack_path = row['pack']
        # MixUp Augmentation with probability 10%
        if random.randint(1, 100) <= 10:
            rand = random.randint(0, len(self) - 1)
            alpha = random.randint(1, 254) / 255
            first_image = np.load(pack_path)
            second_image = np.load(self.csv_file.iloc[rand]['pack'])
            r, g, b, nir, vpm, ed, nd, ps, wc, ww, dr, sd, dp, wa = MixUp()(first_image, second_image, alpha)
        else:
            r, g, b, nir, vpm, ed, nd, ps, wc, ww, dr, sd, dp, wa = np.load(pack_path)
        bg = 1 - np.max((ed, nd, ps, wc, ww, dr, sd, dp, wa), axis=0)
        image = np.array((r, g, b, nir)).transpose((1, 2, 0))
        labels = np.array((bg, dp, dr, ed, nd, ps, wa, ww, wc)).transpose((1, 2, 0))
        # labels = np.array((cs, dp, ps, sw, ww, wc)).transpose((1, 2, 0))

        # intentionally commented
        if self.augmentations:
            image, labels, vpm = self.augmentations(image, labels, vpm)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.mask_transforms:
            labels = self.mask_transforms(labels)

        image_name = os.path.basename(pack_path)
        meta = {
            'valid_pixels_mask': vpm.astype(np.float32),
            'image_id': image_name,
            'mask_id': image_name
        }

        return image, labels, meta

    def __len__(self):
        return self.csv_file.shape[0]




class AgroVision2021Dataset(AgroSegmentationDataset):
    """AgroVision Dataset"""

    def __init__(self, csv_file, image_transforms=None, target_transforms=None, augmentations=None,
                 labels=None):
        self.csv_file = pd.read_csv(csv_file)
        self.image_transforms = image_transforms
        self.mask_transforms = target_transforms
        self.augmentations = augmentations
        self.__labels = np.load(labels)

    def __getitem__(self, item):
        row = self.csv_file.iloc[item]
        pack_path = row['pack']
        # MixUp Augmentation with probability 10%
        if random.randint(1, 100) <= 10:
            rand = random.randint(0, len(self) - 1)
            alpha = random.randint(1, 254) / 255
            first_image = np.load(pack_path)
            second_image = np.load(self.csv_file.iloc[rand]['pack'])
            r, g, b, nir, vpm, sw, cs, ps, wc, ww, dp = MixUp()(first_image, second_image, alpha)
        else:
            r, g, b, nir, vpm, sw, cs, ps, wc, ww, dp = np.load(pack_path)
        bg = 1 - np.max((cs, dp, ps, sw, ww, wc), axis=0)
        image = np.array((r, g, b, nir)).transpose((1, 2, 0))
        labels = np.array((bg, cs, dp, ps, sw, ww, wc)).transpose((1, 2, 0))
        # labels = np.array((cs, dp, ps, sw, ww, wc)).transpose((1, 2, 0))

        # intentionally commented
        if self.augmentations:
            image, labels, vpm = self.augmentations(image, labels, vpm)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.mask_transforms:
            labels = self.mask_transforms(labels)

        image_name = os.path.basename(pack_path)
        meta = {
            'valid_pixels_mask': vpm.astype(np.float32),
            'image_id': image_name,
            'mask_id': image_name
        }

        return image, labels, meta

    def __len__(self):
        return self.csv_file.shape[0]













class AgroVision2021DatasetTest(AgroSegmentationDataset):
    """AgroVision Dataset"""

    def __init__(self, csv_file, image_transforms=None, target_transforms=None, augmentations=None,
                 labels=None):
        self.csv_file = pd.read_csv(csv_file)
        self.image_transforms = image_transforms
        self.mask_transforms = target_transforms
        self.augmentations = augmentations
        self.__labels = np.load(labels)

    def __getitem__(self, item):
        row = self.csv_file.iloc[item]
        pack_path = row['pack']
        r, g, b, nir, vpm, = np.load(pack_path)

        image = np.array((r, g, b,nir)).transpose((1, 2, 0))
        # intentionally commented
        if self.augmentations:
            image, vpm = self.augmentations(image, None, vpm)
        #            image, labels, vpm = self.augmentations(image, labels, vpm)

        if self.image_transforms:
            image = self.image_transforms(image)

        image_name = os.path.basename(pack_path)
        meta = {
            'valid_pixels_mask': vpm.astype(np.float32),
            'image_id': image_name,
            'mask_id': image_name
        }

        return image, meta

    def __len__(self):
        return self.csv_file.shape[0]


class AgroVision2021ClassificationDataset(AgroSegmentationDataset):
    """AgroVision Dataset"""

    def __init__(self, csv_file, image_transforms=None, target_transforms=None, augmentations=None,
                 labels=None):
        self.csv_file = pd.read_csv(csv_file)
        self.image_transforms = image_transforms
        self.mask_transforms = target_transforms
        self.augmentations = augmentations
        self.__labels = np.load(labels)

    def __getitem__(self, item):
        row = self.csv_file.iloc[item]
        pack_path = row['pack']
        r, g, b, nir, vpm, sw, cs, ps, wc, ww, dp = np.load(pack_path)
        image = np.array((r, g, b)).transpose((1, 2, 0))
        bg = 1 - np.max((cs, dp, ps, sw, ww, wc), axis=0)
        labels = np.array((bg, cs, dp, ps, sw, ww, wc)).transpose((1, 2, 0))
        # labels = np.array((cs, dp, ps, sw, ww, wc)).transpose((1, 2, 0))

        # intentionally commented
        if self.augmentations:
            # TODO Rewrite without cycle
            #   image, labels, vpm = self.augmentations(image, labels, vpm)
            image, labels, vpm = self.augmentations(image, labels, vpm)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.mask_transforms:
            labels = self.mask_transforms(labels)

        image_name = os.path.basename(pack_path)
        meta = {
            'valid_pixels_mask': vpm.astype(np.float32),
            'image_id': image_name,
            'mask_id': image_name
        }

        return image, labels, meta

    def __len__(self):
        return self.csv_file.shape[0]


class DataLoader(TorchDataLoader):
    @classmethod
    def from_config(cls, config, dataset, sampler):
        print("WORKERS",config.num_workers)
        return cls(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            sampler=sampler,
            num_workers=config.num_workers
        )
