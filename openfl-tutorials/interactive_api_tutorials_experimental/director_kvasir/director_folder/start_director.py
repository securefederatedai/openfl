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
    asyncio.run(serve(sample_shape=list(sample_shape), target_shape=list(target_shape),
                      disable_tls=False,
                      root_ca='/home/radionov/fork/pki/openfl-tutorials/interactive_api_tutorials_experimental/director_kvasir/director_folder/cert/root_ca.crt',
                      key='/home/radionov/fork/pki/openfl-tutorials/interactive_api_tutorials_experimental/director_kvasir/director_folder/cert/localhost.key',
                      cert='/home/radionov/fork/pki/openfl-tutorials/interactive_api_tutorials_experimental/director_kvasir/director_folder/cert/localhost.crt'))
