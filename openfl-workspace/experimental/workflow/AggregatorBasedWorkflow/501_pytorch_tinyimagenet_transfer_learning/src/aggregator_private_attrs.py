# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torchvision


def aggregator_private_attrs():
    # Load the pre-trained model weights from a file. For example:
    # we have used pre-trained weights from torchvision
    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT,
        progress=True
    )

    return {
        'pretrained_state_dict': model.state_dict()
    }
