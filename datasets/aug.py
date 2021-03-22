from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, Rotate,
    RandomScale, RandomContrast, RandomBrightness, RandomBrightnessContrast, OneOf, Compose, RandomGamma, VerticalFlip,
    HorizontalFlip, PadIfNeeded, RandomCrop, RGBShift
)
from torchvision import datasets, transforms
import albumentations as albu
import cv2


class StandartAugmentation:

    def __init__(self, p=0.8, width=512, height=512):
        self.p = p
        self.width, self.height = width, height
        self.aug = self.__build_augmentator()

    def __call__(self, img, mask, valid_mask):
        augm_res = self.aug(image=img, mask=mask,valid_mask=valid_mask)
        return augm_res['image'], augm_res['mask'], augm_res['valid_mask']

    def __build_augmentator(self):
        return Compose([
            # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, p=0.3),
            # OneOf([
            #     RandomScale(scale_limit=0.05, interpolation=1, p=0.5),
            #     Rotate(limit=7, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5)
            # ], p=0.5),
            # PadIfNeeded(always_apply=True, min_width=self.width, min_height=self.height),
            # RGBShift(p=0.3),
            # RandomCrop(width=self.width, height=self.height),
            OneOf([
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
            ], p=0.5)
            # OneOf([
            #     # RandomBrightness(limit=0.2, always_apply=False, p=0.5),
            #     # RandomContrast(),
            #     RandomGamma()
            # ], p=0.7),
        ], p=self.p, additional_targets={"valid_mask": "mask"})

    class FrogAugmentation:
        def __init__(self, p=0.8, size=(256, 256)):
            self.p = p
            self.width, self.height = size
            self.aug = self.__build_augmentator()

        def __call__(self, x, y):
            augm_res = self.aug(image=x, mask=y)
            return augm_res['image'], augm_res['mask']

        def __build_augmentator(self):
            return Compose([
                ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=13,
                                 interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT,
                                 mask_value=0, value=0.0, p=0.95),
                VerticalFlip(p=0.5),
                RandomContrast(limit=0.4, p=0.95),
            ], )

    class BasicAugmentation:
        def __init__(self):
            self.aug = albu.Compose([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomBrightnessContrast(p=0.6),
                albu.RandomGamma(),
                #     albu.CLAHE()
                albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
                albu.ShiftScaleRotate(),
                albu.Resize(512, 512, always_apply=True),
            ], additional_targets={"valid_mask": "mask"})

        def __call__(self, img, mask, valid_mask):
            res = self.aug(image=img, mask=mask, valid_mask=valid_mask)
            return res["image"], res["mask"], res["valid_mask"]

    class HardAugmentation:
        def __init__(self):
            self.aug = albu.Compose([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomBrightnessContrast(p=0.9),
                albu.RandomGamma(gamma_limit=(60, 140)),
                # albu.CLAHE(),
                # albu.OneOf([
                #     albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                #     albu.GridDistortion(),
                #     albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                # ], p=0.3),
                albu.ShiftScaleRotate(rotate_limit=90, shift_limit=0.15, scale_limit=0.25, p=0.7),
                albu.Resize(512, 512, always_apply=True),
            ], additional_targets={"valid_mask": "mask"})

        def __call__(self, img, mask, valid_mask):
            res = self.aug(image=img, mask=mask, valid_mask=valid_mask)
            return res["image"], res["mask"], res["valid_mask"]

    class HardSpatial:
        def __init__(self):
            self.aug = albu.Compose([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                # albu.RandomBrightnessContrast(p=0.6),
                # albu.RandomGamma(gamma_limit=(60, 140)),
                # albu.CLAHE(),
                # albu.OneOf([
                #     albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                #     albu.GridDistortion(),
                #     albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                # ], p=0.3),
                albu.ShiftScaleRotate(rotate_limit=90, shift_limit=0.15, scale_limit=0.25, p=0.7),
                albu.Resize(512, 512, always_apply=True),
            ], additional_targets={"valid_mask": "mask"})

        def __call__(self, img, mask, valid_mask):
            res = self.aug(image=img, mask=mask, valid_mask=valid_mask)
            return res["image"], res["mask"], res["valid_mask"]

    class HardColour:
        def __init__(self):
            self.aug = albu.Compose([
                # albu.HorizontalFlip(),
                # albu.VerticalFlip(),
                albu.RandomBrightnessContrast(p=0.9),
                albu.RandomGamma(gamma_limit=(60, 140)),
                # albu.CLAHE(),
                # albu.OneOf([
                #     albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                #     albu.GridDistortion(),
                #     albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                # ], p=0.3),
                # albu.ShiftScaleRotate(rotate_limit=90, shift_limit=0.15, scale_limit=0.25, p=0.7),
                # albu.Resize(512, 512, always_apply=True),
            ], additional_targets={"valid_mask": "mask"})

        def __call__(self, img, mask, valid_mask):
            res = self.aug(image=img, mask=mask, valid_mask=valid_mask)
            return res["image"], res["mask"], res["valid_mask"]

    class Retina:
        def __init__(self):
            self.aug = albu.Compose([
                albu.Resize(576, 576),
                albu.IAAAffine(
                    rotate=(-180, 180),
                    scale=(0.8889, 1.0),
                    shear=(-36, 36)),
                albu.CenterCrop(512, 512),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomContrast((0.9, 1.1))
            ], additional_targets={"valid_mask": "mask"})

        def __call__(self, img, mask, valid_mask):
            res = self.aug(image=img, mask=mask, valid_mask=valid_mask)
            return res["image"], res["mask"], res["valid_mask"]
