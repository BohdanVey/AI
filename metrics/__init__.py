import segmentation_models_pytorch as smp

from .binary_classification_meter import BinaryClassificationMeter
from .binary_segmentation_meter import AverageMetricsMeter
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics


def make_metrics(config):
    return [
        getattr(smp.utils.metrics, metric_type)(**(config.params if config.params else {}))
        for metric_type, config in config.items()
    ]


def calculate_iou(target, output):
    # TODO REWRITE THIS
    out = nn.Sigmoid()(output).detach()
    ans = out > 0.5
    intersection = torch.sum(torch.sum(torch.sum(ans * target, dim=0), dim=1), dim=1)
    union = torch.sum(torch.sum(torch.sum(ans, dim=0), dim=1), dim=1) + torch.sum(
        torch.sum(torch.sum(target, dim=0), dim=1), dim=1) - intersection
    return intersection.cpu().numpy().astype("int64"), union.cpu().numpy().astype("int64")





def calculate_confusion_matrix(target, output):
    target_matrix = torch.argmax(target, dim=1).cpu()
    output_matrix = torch.argmax(output, dim=1).detach().cpu()
    return sklearn.metrics.confusion_matrix(target_matrix.reshape(-1), output_matrix.reshape(-1), labels=range(7))
