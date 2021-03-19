from torch import nn
from torch.functional import F


class BinaryCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert(logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        return loss.mean()
