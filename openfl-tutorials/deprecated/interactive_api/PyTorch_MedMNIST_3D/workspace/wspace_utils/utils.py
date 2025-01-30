# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import numpy as np


class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):

        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()

        return voxel.astype(np.float32)


def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model


def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children():
        if (
            hasattr(nn, child.__class__.__name__)
            and 'batchnorm' in child.__class__.__name__.lower()
        ):
            TargetClass = globals()['Synchronized' + child.__class__.__name__]  # noqa: N806
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)
