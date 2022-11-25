# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import time
import sys
import shutil
import subprocess

from openfl.utilities.workspace import set_directory


CA_PATH = Path('~/.local/ca').expanduser()
CA_URL = 'localhost:9123'
CA_PASSWORD = 'qwerty'


if __name__ == '__main__':
    shutil.rmtree(CA_PATH)
    # 1. Install CA
    subprocess.check_call([
        'fx', 'pki', 'install'
        '-p', str(CA_PATH),
        '--password', str(CA_PASSWORD)
    ])

    # 2. Start CA server
    ca_server = subprocess.Popen([
        'fx', 'pki', 'run',
        '-p', str(CA_PATH)
    ])
    time.sleep(3)  # Wait for server to initialize

    if ca_server.poll() is not None:
        print('CA Server failed to initialize')
        sys.exit(1)

    try:
        # 3. Generate token for Director
        token_output = subprocess.check_output([
            'fx', 'pki', 'get-token',
            '-n', 'localhost'
            '-p', str(CA_PATH)
        ])
        token = token_output.decode('utf-8').split('\n')[1]

        # 4. Certify the Director with generated token
        subprocess.check_call([
            'fx', 'pki', 'certify',
            '-n', 'localhost'
            '-t', token,
            '-c', f'{CA_PATH / "cert"}'
            '-p', str(CA_PATH)
        ])

        # 5. Start the Director with TLS enabled
        director_config_path = Path(
            'tests/github/interactive_api_director/experiments/pytorch_kvasir_unet/'
            'director/director_config.yaml'
        )
        root_cert_path = CA_PATH / 'cert' / 'root_ca.crt'
        private_key_path = CA_PATH / 'cert' / 'localhost.key'
        public_cert_path = CA_PATH / 'cert' / 'localhost.crt'
        director = subprocess.Popen([
            'fx', 'director', 'start',
            '-c', str(director_config_path),
            '-rc', str(root_cert_path),
            '-pk', str(private_key_path),
            '-oc', str(public_cert_path)
        ])
        time.sleep(3)  # Wait for Director to start

        # 6. Start the Envoy without TLS
        expected_error = ('Handshake failed with fatal error SSL_ERROR_SSL: error:100000f7:'
                          'SSL routines:OPENSSL_internal:WRONG_VERSION_NUMBER.')
        try:
            envoy_folder = ('tests/github/interactive_api_director/'
                            'experiments/pytorch_kvasir_unet/envoy/')
            with set_directory(envoy_folder):
                envoy = subprocess.Popen([
                    'fx', 'envoy', 'start',
                    '-n', 'one',
                    '-dh', 'localhost',
                    '-dp', '50051',
                    '--disable-tls',
                    '-ec', 'envoy_config.yaml'
                ])
                time.sleep(10)
                # Expectation is that secure server won't let insecure client connect
                if envoy.poll() != 1 or expected_error not in envoy.communicate():
                    envoy.kill()
                    sys.exit(1)
        finally:
            director.kill()
    finally:
        ca_server.kill()
        shutil.rmtree(CA_PATH)
