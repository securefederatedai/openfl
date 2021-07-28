# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Start director module."""

import logging

from openfl.component.director import Director
from openfl.interface.cli import setup_logging
from openfl.transport import DirectorGRPCServer

setup_logging()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    sample_shape = ['529', '622', '3']
    target_shape = ['529', '622']
    director = DirectorGRPCServer(
        director_cls=Director,
        sample_shape=sample_shape,
        target_shape=target_shape,
        disable_tls=False,
        # an absolute path is required
        root_ca='./cert/root_ca.crt',
        key='./cert/localhost.key',
        cert='./cert/localhost.crt'
    )
    director.start()
