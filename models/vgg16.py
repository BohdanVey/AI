import torch.nn as nn
from torchvision import models


def get_vgg16(pretrained, num_classes): # noqa
    model = models.vgg16(pretrained=pretrained)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    return model
