# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""openfl.pipelines module."""

import pkgutil

if pkgutil.find_loader('torch'):
    from .eden_pipeline import EdenPipeline  # NOQA

from .kc_pipeline import KCPipeline
from .no_compression_pipeline import NoCompressionPipeline
from .random_shift_pipeline import RandomShiftPipeline
from .skc_pipeline import SKCPipeline
from .stc_pipeline import STCPipeline
from .tensor_codec import TensorCodec

__all__ = [
    'NoCompressionPipeline',
    'RandomShiftPipeline',
    'STCPipeline',
    'SKCPipeline',
    'KCPipeline',
    'EdenPipeline',
    'TensorCodec',
]
