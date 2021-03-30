import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class BinaryCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert (logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        return loss.mean()


class BinaryLogDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def __call__(self, input, target):
        input = torch.sigmoid(input)
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return - ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).log()


class WrappedBinaryLogDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def cl_run(self, input, target):
        iflat = input.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()
        return - ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).log()

    def forward(self, input, target, mask):
        input = torch.sigmoid(input * mask.unsqueeze(1))
        v = None
        for cl in range(input.size(1)):
            if v is None:
                v = self.cl_run(input[:, cl], target[:, cl])
            else:
                v = v + self.cl_run(input[:, cl], target[:, cl])
        return v.sum()




class ImageBinaryLogDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def __call__(self, input, target):
        input = torch.sigmoid(input)
        batch = input.shape[0]
        iflat = input.view(batch, -1)
        tflat = target.view(batch, -1)
        intersection = (iflat * tflat).sum(-1)
        return - ((2.0 * intersection + self.smooth) / (iflat.sum(-1) + tflat.sum(-1) + self.smooth)).log().mean()


class OfficialBinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(OfficialBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.float()
        pos = (targets > 0.5).float()
        neg = (targets < 0.5).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        F_loss = self.alpha * pos * F_loss + (1 - self.alpha) * neg * F_loss
        return torch.mean(F_loss)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0.75, reduce=None):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if self.reduce:
            return loss.mean()  # (B*C*W*H)
        return loss


class FixedBinaryFocalWrapped(nn.Module):
    def __init__(self, gamma=0.75):
        super().__init__()
        self.loss = BinaryFocalLoss(gamma)

    def forward(self, logit, truth, mask):
        loss = self.loss.forward(logit, truth)
        pixelnum = mask.sum()
        # pixelnum[pixelnum == 0] = 1.0
        pixelnum = torch.clamp(pixelnum, min=1.0)
        loss = (loss * mask.unsqueeze(1)).sum() / pixelnum
        return loss


class BinaryFocalWrappedSpecGrad(nn.Module):
    def __init__(self, gamma=0.75):
        super().__init__()
        self.loss = BinaryFocalLoss(gamma)

    def forward(self, logit, truth, mask):
        loss = self.loss.forward(logit, truth)
        pixelnum = mask.sum(-1).sum(-1)
        should_use_grad = (truth.sum(-1).sum(-1) > 0).float()
        # pixelnum = (pixelnum.unsqueeze(1) * should_use_grad)
        pixelnum = torch.clamp(pixelnum, min=1.0)
        loss = (loss * mask.unsqueeze(1))
        loss = (loss * should_use_grad.unsqueeze(-1).unsqueeze(-1)).sum(-1).sum(-1).sum(-1) / pixelnum
        return loss.mean()


class OnlyPositiveBinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.sum() / ((target > 0.5).float().sum() + 1.0)


class BinaryFocalDice(nn.Module):
    def __init__(self, alpha=10.0, gamma=0.75, smooth=1.0, divider=1.0):
        super().__init__()
        self.alpha = alpha
        self.focal = BinaryFocalLoss(gamma)
        self.log_dice_loss = BinaryLogDice(smooth=smooth)
        self.divider = divider
        print("Using divider", self.divider)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) + self.log_dice_loss(input, target)
        return loss / self.divider


class BinaryOnlyPositiveFocalDice(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smooth=1.0, divider=1.0):
        super().__init__()
        self.alpha = alpha
        self.focal = OnlyPositiveBinaryFocalLoss(gamma)
        self.log_dice_loss = BinaryLogDice(smooth=smooth)
        self.divider = divider
        print("Using divider", self.divider)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) + self.log_dice_loss(input, target)
        return loss / self.divider


class BinaryCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert (logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        return loss.mean()


class FixedMultiLabelCrossEntropyMask(nn.Module):
    def forward(self, logit, truth, mask):
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        pixelnum = mask.sum()
        # pixelnum[pixelnum == 0] = 1.0
        pixelnum = torch.clamp(pixelnum, min=1.0)
        loss = (loss * mask.unsqueeze(1)).sum() / pixelnum
        return loss


class MultiLabelCrossEntropyMask(nn.Module):
    def forward(self, logit, truth, mask):
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        pixelnum = mask.sum(-1).sum(-1)
        # pixelnum[pixelnum == 0] = 1.0
        pixelnum = torch.clamp(pixelnum, min=1.0)
        loss = (loss * mask.unsqueeze(1)).sum(-1).sum(-1).sum(-1) / pixelnum
        return loss.mean()


class FixedMultiLabelCrossEntropyMaskGradSpec(nn.Module):
    def forward(self, logit, truth, mask):
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        pixelnum = mask.sum((1, 2)).unsqueeze(-1)  # res -> (B, 1)
        should_use_grad = (truth.sum((2, 3)) > 0).float()
        pixelnum = (pixelnum * should_use_grad).sum(0)  # res -> (C)
        pixelnum = torch.clamp(pixelnum, min=1.0)
        loss = (loss * mask.unsqueeze(1))
        loss = (loss * should_use_grad.unsqueeze(-1).unsqueeze(-1)).sum((0, 2, 3)) / pixelnum
        return loss.sum()


class MultiLabelCrossEntropyMaskGradSpec(nn.Module):
    def forward(self, logit, truth, mask):
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        pixelnum = mask.sum(-1).sum(-1)
        should_use_grad = (truth.sum(-1).sum(-1) > 0).float()
        # pixelnum = (pixelnum.unsqueeze(1) * should_use_grad)
        pixelnum = torch.clamp(pixelnum, min=1.0)
        loss = (loss * mask.unsqueeze(1))
        loss = (loss * should_use_grad.unsqueeze(-1).unsqueeze(-1)).sum(-1).sum(-1).sum(-1) / pixelnum
        return loss.mean()


class BinaryBatchWeightedCrossEntropy(nn.Module):
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        assert (logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        if 0:
            loss = loss.mean()

        if 1:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()

        return loss


class WeightedLogDice(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        """
        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = 1.0
        self.gamma = gamma
        sum = self.beta + self.alpha
        if sum != 1:
            self.beta = self.beta / sum
            self.alpha = self.alpha / sum

    # @staticmethod
    def forward(self, pred, target):
        target = target.view(-1).float()
        pred = pred.sigmoid().view(-1)
        # _, input_label = input.max(1)
        true_pos = (target * pred).sum()
        false_neg = (target * (1 - pred)).sum()
        false_pos = ((1 - target) * pred).sum()

        loss = (2 * true_pos + self.smooth) / (
                2 * true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)

        return - loss.log()


class LogDiceAndFocalLoss(nn.Module):
    def __init__(self, dice_weight=2.0, focal_weight=4.0):
        super().__init__()
        self.logdice = WeightedLogDice()
        self.focal = FixedBinaryFocalWrapped()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, input, target, mask):
        return self.logdice(input, target) * self.dice_weight + self.focal_weight * self.focal(input, target,
                                                                                                     mask)















class ACW_loss(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5, ignore_index=255):
        super(ACW_loss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target, valid_mask = None):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        pred = F.softmax(prediction, 1)
        answer = np.zeros((target.shape[0], target.shape[2], target.shape[3]))
        for i in range(target.shape[0]):
            answer[i, :, :] = np.argmax(target[i, :, :, :].cpu().numpy(), axis=0)
        answer = torch.from_numpy(answer.astype(np.int64)).to(target.device)

        one_hot_label, mask = self.encode_one_hot_label(pred, answer)
        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        # one = torch.ones_like(err)
        # TODO REMOVE POWER
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None