# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import socket
import subprocess
import os


def test_aggregator_common_name(common_name, should_fail):
    exit_code = subprocess.call([
        'fx', 'aggregator', 'generate-cert-request',
        '--fqdn', common_name
    ])
    if should_fail != bool(exit_code):
        raise Exception(f'Aggregator CSR generation command finished with exit code {exit_code}. '
                        f'Common Name: "{common_name}".')


if __name__ == '__main__':
    prefix = 'fed_workspace'
    subprocess.check_call([
        'fx', 'workspace', 'create',
        '--prefix', str(prefix),
        '--template', 'keras_cnn_mnist'
    ])
    os.chdir(prefix)
    subprocess.check_call(['fx', 'workspace', 'certify'])

    # invalid common name
    test_aggregator_common_name('not_an_fqdn', should_fail=True)

    # case when CA is located on separate machine
    test_aggregator_common_name('some.fqdn.com', should_fail=False)

    # valid common name
    test_aggregator_common_name(socket.getfqdn(), should_fail=False)
