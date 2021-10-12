"""Data transform functions."""

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


def resize(image, shape=(256, 256)):
    """Resize image."""
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def bilinears(images, shape) -> np.ndarray:
    """Generate binlinears."""
    import cv2
    n = images.shape[0]
    new_shape = (n,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(n):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret


def gray2rgb(images):
    """Change gray to rgb images."""
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    return images


def detection_auroc(obj, anomaly_scores, labels):
    """detection auroc calculation."""
    # 1: anomaly 0: normal
    auroc = roc_auc_score(labels, anomaly_scores)
    return auroc


def segmentation_auroc(obj, anomaly_maps, masks):
    """segmentation auroc calculation."""
    gt = masks
    gt = gt.astype(np.int32)
    gt[gt == 255] = 1  # 1: anomaly

    anomaly_maps = bilinears(anomaly_maps, (256, 256))
    auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
    return auroc
