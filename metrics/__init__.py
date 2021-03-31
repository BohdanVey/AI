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
    out = nn.Sigmoid()(output).detach().cpu().numpy()
    tar = target.cpu().numpy()
    ans = out > 0.5
    intersection = np.sum(np.sum(np.sum(ans * tar, axis=0), axis=1), axis=1)
    union = np.sum(np.sum(np.sum(ans, axis=0), axis=1), axis=1) + np.sum(
        np.sum(np.sum(tar, axis=0), axis=1), axis=1) - intersection
    return intersection.astype("int64"), union.astype("int64")


def calculate_iou6(target, output):
    out = nn.Sigmoid()(output).detach().cpu().numpy()
    tar = target.cpu().numpy()
    ans = out > 0.6

    background = 1 - np.max(tar, axis=1)
    background = background.reshape((tar.shape[0], 1, tar.shape[2], tar.shape[3]))
    tar = np.concatenate((background, tar), axis=1)
    background = 1 - np.max(ans, axis=1)
    background = background.reshape((ans.shape[0], 1, ans.shape[2], ans.shape[3]))
    ans = np.concatenate((background, ans), axis=1)
    intersection = np.sum(np.sum(np.sum(ans * tar, axis=0), axis=1), axis=1)
    union = np.sum(np.sum(np.sum(ans, axis=0), axis=1), axis=1) + np.sum(
        np.sum(np.sum(tar, axis=0), axis=1), axis=1) - intersection
    return intersection.astype("int64"), union.astype("int64")


def calculate_confusion_matrix(target, output):
    target_matrix = torch.argmax(target, dim=1).cpu()
    output_matrix = torch.argmax(output, dim=1).detach().cpu()

    return sklearn.metrics.confusion_matrix(target_matrix.reshape(-1), output_matrix.reshape(-1),labels=range(7))
