# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data transform functions."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score


def bilinears(images, shape) -> np.ndarray:
    """Generate binlinears."""
    import cv2
    n = images.shape[0]
    new_shape = (n,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(n):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret


def bal_acc_score(obj, predictions, labels):
    """Calculate balanced accuracy score."""
    precision, recall, thresholds = precision_recall_curve(labels.flatten(), predictions.flatten())
    f1_score = (2 * precision * recall) / (precision + recall)
    threshold = thresholds[np.argmax(f1_score)]
    prediction_result = predictions > threshold
    ba_score = balanced_accuracy_score(labels, prediction_result)
    return ba_score


def detection_auroc(obj, anomaly_scores, labels):
    """Calculate detection auroc."""
    # 1: anomaly 0: normal
    auroc = roc_auc_score(labels, anomaly_scores)
    return auroc


def segmentation_auroc(obj, anomaly_maps, masks):
    """Calculate segmentation auroc."""
    gt = masks
    gt = gt.astype(np.int32)
    gt[gt == 255] = 1  # 1: anomaly

    anomaly_maps = bilinears(anomaly_maps, (256, 256))
    auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
    return auroc
