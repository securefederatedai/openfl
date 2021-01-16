# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NoCompressionPipeline module."""

from .pipeline import TransformationPipeline, Float32NumpyArrayToBytes


class NoCompressionPipeline(TransformationPipeline):
    """The data pipeline without any compression."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(NoCompressionPipeline, self).__init__(
            transformers=[Float32NumpyArrayToBytes()], **kwargs)
