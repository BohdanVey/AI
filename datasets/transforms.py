import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor # noqa


class RGBOnly:
    def __call__(self, image):
        b, g, r, nir = image
        return np.array((b, g, r))


class ToNumpyInt32:
    def __call__(self, image):
        return image.astype(np.int32)


class FromNumpy:
    def __call__(self, image):
        return torch.from_numpy(image)


class ToTorchFloat:
    def __call__(self, image):
        return image.float()


class ToTorchLong:
    def __call__(self, image):
        return image.long()


class ChannelsFirst:
    def __call__(self, image):
        h, w, c = 0, 1, 2
        return image.transpose((c, h, w))


class ChannelsLast:
    def __call__(self, image):
        c, h, w = 0, 1, 2
        return image.transpose((h, w, c))