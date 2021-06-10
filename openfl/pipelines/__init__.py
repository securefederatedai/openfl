# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl.pipelines module."""

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
    'TensorCodec',
]
