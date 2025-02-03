# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(output, target):
    """Dice loss function."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def soft_dice_coef(output, target):
    """Dice coef metric function."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    return score.sum()


class DoubleConv(nn.Module):
    """Convolutions with BN and activation."""

    def __init__(self, in_ch, out_ch):
        """Initialize.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels

        """
        super(DoubleConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Run forward."""
        x = self.conv(x)
        return x


class Down(nn.Module):
    """UNet downscaling. MaxPool with double convolution."""

    def __init__(self, in_ch, out_ch):
        """Initialize.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels

        """
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        """Run forward."""
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    """UNet upscaling."""

    def __init__(self, in_ch, out_ch):
        """Initialize.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels

        """
        super(Up, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """Run forward."""
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
