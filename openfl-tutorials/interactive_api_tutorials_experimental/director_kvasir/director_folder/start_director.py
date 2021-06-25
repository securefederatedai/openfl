# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Start director module."""


import asyncio
import logging

from openfl.component.director.director import serve

from openfl.interface.cli import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    sample_shape = ['529', '622', '3']
    target_shape = ['529', '622']
    asyncio.run(serve(sample_shape=list(sample_shape), target_shape=list(target_shape)))
