# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""NoCompressionPipeline module."""

from openfl.pipelines.pipeline import Float32NumpyArrayToBytes, TransformationPipeline


class NoCompressionPipeline(TransformationPipeline):
    """The data pipeline without any compression."""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(transformers=[Float32NumpyArrayToBytes()], **kwargs)
