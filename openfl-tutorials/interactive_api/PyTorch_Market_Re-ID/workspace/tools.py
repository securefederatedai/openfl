# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tools for metric computation and Dataloader."""

import copy
import random
from collections import defaultdict
from logging import getLogger

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

logger = getLogger(__name__)


class AverageMeter:
    """
    Computes and stores the average and current value.

    Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        """Initialize Average Meter."""
        self.reset()

    def reset(self):
        """Reset values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update values."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_ap_cmc(index, good_index, junk_index):
    """Compute validation metrics."""
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    """Evaluate model."""
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)  # from small to large

    num_no_gt = 0  # num of query imgs without groundtruth
    num_r1 = 0
    cmc = np.zeros(len(g_pids))
    ap = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, cmc_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if cmc_tmp[0] == 1:
            num_r1 += 1
        cmc = cmc + cmc_tmp
        ap += ap_tmp

    if num_no_gt > 0:
        logger.error(f'{num_no_gt} query imgs do not have groundtruth.')

    cmc = cmc / (num_q - num_no_gt)
    mean_ap = ap / (num_q - num_no_gt)

    return cmc, mean_ap


@torch.no_grad()
def extract_feature(model, dataloader):
    """Extract features for validation."""
    features, pids, camids = [], [], []
    for imgs, (batch_pids, batch_camids) in dataloader:
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data
        batch_features_flip = model(flip_imgs).data
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, pids, camids


def fliplr(img):
    """Flip horizontal."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)

    return img_flip


class RandomIdentitySampler(Sampler):
    """
    Random Sampler.

    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances=4):
        """Initialize Sampler."""
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, (pid, _)) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        """Iterate over Sampler."""
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        """Return number of examples in an epoch."""
        return self.length
