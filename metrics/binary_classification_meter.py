import torch


class BinaryClassificationMeter:
    def __init__(self, device):
        self.conf_mat = torch.zeros(2, 2).to(device)
        self.running_corrects = torch.tensor(0.0, dtype=torch.double)
        self.total_count = torch.tensor(0)

    def update(self, true, pred):
        tp = ((pred == 1) & (true == 1)).sum()
        fp = ((pred == 1) & (true == 0)).sum()
        fn = ((pred == 0) & (true == 1)).sum()
        tn = ((pred == 0) & (true == 0)).sum()

        self.conf_mat += torch.stack([tp, fp, fn, tn]).reshape(2, 2)
        self.running_corrects += (tp + tn)
        self.total_count += true.size(0)  # batch size

    def get_metrics(self):
        tp, fp, fn, tn = self.conf_mat.flatten()

        accuracy = self.running_corrects / self.total_count
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics = {
            'conf_mat': self.conf_mat.cpu().numpy(),
            'accuracy': accuracy.cpu().numpy().item(),
            'precision': precision.cpu().numpy().item(),
            'recall': recall.cpu().numpy().item(),
            'f1': f1.cpu().numpy().item()
        }

        return metrics
