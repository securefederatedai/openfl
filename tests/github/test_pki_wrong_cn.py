# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import socket
import os


PREFIX = Path('~/.local/workspace').expanduser()
TEMPLATE = 'keras_cnn_mnist'


def test_aggregator_common_name(common_name, should_fail):
    exit_code = os.system(f'fx aggregator generate-cert-request --fqdn {common_name}')
    if should_fail != bool(exit_code):
        raise Exception(f'Aggregator certificate generation command for {common_name} '
                        f'finished with exit code {exit_code}.')


if __name__ == '__main__':
    os.system(f'fx workspace create --prefix {PREFIX} --template {TEMPLATE}')
    os.chdir(PREFIX)
    os.system('fx workspace certify')

    test_aggregator_common_name('not_an_fqdn', should_fail=True)
    test_aggregator_common_name('some.fqdn.com', should_fail=True)
    test_aggregator_common_name(socket.getfqdn(), should_fail=False)
