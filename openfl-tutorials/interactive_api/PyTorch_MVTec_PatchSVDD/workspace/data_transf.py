import argparse
import torch
from functools import reduce
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os, shutil
import _pickle as p
from contextlib import contextmanager

def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def bilinears(images, shape) -> np.ndarray:
    import cv2
    N = images.shape[0]
    new_shape = (N,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(N):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret


def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    return images

def detection_auroc(obj, anomaly_scores, labels):
    # 1: anomaly 0: normal
    auroc = roc_auc_score(labels, anomaly_scores)
    return auroc


def segmentation_auroc(obj, anomaly_maps, masks):
    gt = masks
    gt = gt.astype(np.int32)
    gt[gt == 255] = 1  # 1: anomaly

    anomaly_maps = bilinears(anomaly_maps, (256, 256))
    auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
    return auroc
