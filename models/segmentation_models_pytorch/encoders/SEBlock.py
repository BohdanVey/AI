import torch
import torch.nn as nn
import torch.nn.functional as F


class HSigmoid(nn.Module):

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


def get_activation_layer(activation):
    assert (activation is not None)
    return activation()


class SEBlock(nn.Module):

    def __init__(self,
                 channels,
                 device = "cuda:0",
                 reduction=1,
                 approx_sigmoid=False,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(
            in_channels=channels,
            out_channels=mid_cannels,
            bias=True).to(device)
        self.activ = get_activation_layer(activation)
        self.conv2 = conv1x1(
            in_channels=mid_cannels,
            out_channels=channels,
            bias=True).to(device)
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w

        return x
