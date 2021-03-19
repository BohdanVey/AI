import segmentation_models_pytorch as smp

from .binary_classification_meter import BinaryClassificationMeter
from .binary_segmentation_meter import AverageMetricsMeter


def make_metrics(config):
    return [
        getattr(smp.utils.metrics, metric_type)(**(config.params if config.params else {}))
        for metric_type, config in config.items()
    ]
