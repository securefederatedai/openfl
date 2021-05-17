"""Layers for Unet model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(output, target):
    """Calculate loss."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = m1 * m2
    score = 2.0 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def soft_dice_coef(output, target):
    """Calculate soft DICE coefficient."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = m1 * m2
    score = 2.0 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    return score.sum()


class double_conv(nn.Module):
    """Pytorch double conv class."""

    def __init__(self, in_ch, out_ch):
        """Initialize layer."""
        super(double_conv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Do forward pass."""
        x = self.conv(x)
        return x


class down(nn.Module):
    """Pytorch nn module subclass."""

    def __init__(self, in_ch, out_ch):
        """Initialize layer."""
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        """Do forward pass."""
        x = self.mpconv(x)
        return x


class up(nn.Module):
    """Pytorch nn module subclass."""

    def __init__(self, in_ch, out_ch, bilinear=False):
        """Initialize layer."""
        super(up, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_ch, in_ch // 2, 2, stride=2
            )
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        """Do forward pass."""
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX
                   // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
