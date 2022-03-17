"""Utilities."""

import os
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Union

import _pickle as p
import numpy as np
import torch
from torch.utils.data import Dataset


def to_device(obj: Union[torch.Tensor, dict, list, tuple],
              device: str, non_blocking: bool = False) -> Union[torch.Tensor, dict, list, tuple]:
    """Copy to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking=non_blocking)
                for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_device(v, device, non_blocking=non_blocking)
                for v in obj]

    if isinstance(obj, tuple):
        return tuple([to_device(v, device, non_blocking=non_blocking)
                     for v in obj])


@contextmanager
def task(_: Any) -> Iterable:
    """Yield."""
    yield


class DictionaryConcatDataset(Dataset):
    """Concate dictionaries."""

    def __init__(self, d_of_datasets: Dict[str, Dataset]) -> None:
        """Initialize."""
        self.d_of_datasets = d_of_datasets
        lengths = [len(d) for d in d_of_datasets.values()]
        self._length = min(lengths)
        self.keys = self.d_of_datasets.keys()
        assert min(lengths) == max(lengths), 'Length of the datasets should be the same'

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item."""
        return {
            key: self.d_of_datasets[key][idx]
            for key in self.keys
        }

    def __len__(self) -> int:
        """Get length."""
        return self._length


def crop_chw(image: Any, i: int, j: int, k: int, s: int = 1) -> Any:
    """Crop func."""
    if s == 1:
        h, w = i, j
    else:
        h = s * i
        w = s * j
    return image[:, h: h + k, w: w + k]


def cnn_output_size(h: int, k: int, s: int = 1, p: int = 0) -> int:
    """Output size.

    :param int H: input_size
    :param int K: filter_size
    :param int S: stride
    :param int P: padding
    :return:.

    """
    return 1 + (h - k + 2 * p) // s


def crop_image_chw(image: Any, coord: int, k: int) -> Any:
    """Crop func."""
    h, w = coord
    return image[:, h: h + k, w: w + k]


def load_binary(fpath: Any, encoding: str = 'ASCII') -> Any:
    """Load binaries."""
    with open(fpath, 'rb') as f:
        return p.load(f, encoding=encoding)


def save_binary(d: Any, fpath: Any) -> None:
    """Save binary."""
    with open(fpath, 'wb') as f:
        p.dump(d, f)


def makedirpath(fpath: str) -> None:
    """Make path."""
    dpath = os.path.dirname(fpath)
    if dpath:
        os.makedirs(dpath, exist_ok=True)


def distribute_scores(score_masks: Any, output_shape: Any, k: int, s: int) -> np.ndarray:
    """Distribute scores."""
    n_all = score_masks.shape[0]
    results = [distribute_score(score_masks[n], output_shape, k, s) for n in range(n_all)]
    return np.asarray(results)


def distribute_score(score_mask: Any, output_shape: Any, k: int, s: int) -> np.ndarray:
    """Distribute scores."""
    h, w = output_shape
    mask = np.zeros([h, w], dtype=np.float32)
    cnt = np.zeros([h, w], dtype=np.int32)

    i, j = score_mask.shape[:2]
    for i_ in range(i):
        for j_ in range(j):
            h_, w_ = i_ * s, j_ * s

            mask[h_: h_ + k, w_: w_ + k] += score_mask[i_, j_]
            cnt[h_: h_ + k, w_: w_ + k] += 1

    cnt[cnt == 0] = 1

    return mask / cnt
