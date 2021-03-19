from segmentation_models_pytorch.utils.meter import AverageValueMeter


class AverageMetricsMeter:
    def __init__(self, metrics, device):
        self.metrics = metrics
        self.metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
        self._to_device(device)

    def _to_device(self, device):
        for metric in self.metrics:
            metric.to(device)

    def update(self, gt, pr):
        for metric_fn in self.metrics:
            metric_value = metric_fn(pr, gt).cpu().detach().numpy()
            self.metrics_meters[metric_fn.__name__].add(metric_value)

    def get_metrics(self):
        metrics = {k: v.mean for k, v in self.metrics_meters.items()}
        return metrics
