# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compute ArcFace loss and Triplet loss."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceLoss(nn.Module):
    """ArcFace loss."""

    def __init__(self, margin=0.1, scale=16, easy_margin=False):
        """Initialize ArcFace loss."""
        super(ArcFaceLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.easy_margin = easy_margin

    def forward(self, pred, target):
        """Compute forward."""
        # make a one-hot index
        index = pred.data * 0.0  # size = (B, Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.bool()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = pred[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m - sin_t * sin_m

        cond_v = cos_t - math.cos(math.pi - self.m)
        cond = F.relu(cond_v)
        keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)

        output = pred * 1.0  # size = (B, Classnum)
        output[index] = cos_t_add_m
        output = self.s * output

        return F.cross_entropy(output, target)


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
        distance (str): distance for triplet.
    """

    def __init__(self, margin=0.3, distance='cosine'):
        """Initialize Triplet loss."""
        super(TripletLoss, self).__init__()

        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Compute forward.

        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        inputs = F.normalize(inputs, p=2, dim=1)
        dist = - torch.mm(inputs, inputs.t())

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
