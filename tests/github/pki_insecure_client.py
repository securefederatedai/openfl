# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import time
import sys
import shutil
import subprocess
import signal
from openfl.interface.pki import run as run_pki

from openfl.utilities.workspace import set_directory
from threading import Thread


CA_PATH = Path('~/.local/ca').expanduser()
CA_URL = 'localhost:9123'
CA_PASSWORD = 'qwerty'
DIRECTOR_SUBJECT_NAME = 'localhost'
WORKSPACE_PATH = Path('tests/github/interactive_api_director/experiments/pytorch_kvasir_unet/')
EXPECTED_ERROR = ('Handshake failed with fatal error')


def start_ca_server():
    def start_server():
        try:
            run_pki(str(CA_PATH))
        except subprocess.CalledProcessError as e:
            if signal.Signals(-e.returncode) is signal.SIGKILL:  # intended stop signal
                pass
            else:
                raise
    ca_server = Thread(target=start_server, daemon=True)
    ca_server.start()
    time.sleep(3)  # Wait for server to initialize

    if not ca_server.is_alive():
        print('CA Server failed to initialize')
        sys.exit(1)
    return ca_server


def install_ca():
    pki_install = subprocess.Popen([
        'fx', 'pki', 'install',
        '-p', str(CA_PATH),
        '--password', str(CA_PASSWORD)
    ], stdin=subprocess.PIPE)
    time.sleep(3)
    pki_install.communicate('y'.encode('utf-8'))


def get_token():
    token_output = subprocess.check_output([
        'fx', 'pki', 'get-token',
        '-n', DIRECTOR_SUBJECT_NAME,
        '-p', str(CA_PATH)
    ])
    return token_output.decode('utf-8').split('\n')[1]


def certify_director(token):
    subprocess.check_call([
        'fx', 'pki', 'certify',
        '-n', DIRECTOR_SUBJECT_NAME,
        '-t', token,
        '-c', f'{CA_PATH / "cert"}',
        '-p', str(CA_PATH)
    ])


def start_director(tls=True):
    director_config_path = WORKSPACE_PATH / 'director' / 'director_config.yaml'
    root_cert_path = CA_PATH / 'cert' / 'root_ca.crt'
    private_key_path = CA_PATH / 'cert' / f'{DIRECTOR_SUBJECT_NAME}.key'
    public_cert_path = CA_PATH / 'cert' / f'{DIRECTOR_SUBJECT_NAME}.crt'
    params = [
        '-c', str(director_config_path),
        '-rc', str(root_cert_path),
        '-pk', str(private_key_path),
        '-oc', str(public_cert_path)
    ]
    if not tls:
        params.append('--disable-tls')
    director = subprocess.Popen([
        'fx', 'director', 'start',
        *params
    ], stderr=subprocess.PIPE)
    return director


def stop_server(ca_server):
    subprocess.check_call([
        'fx', 'pki', 'uninstall',
        '-p', str(CA_PATH)
    ])
    if ca_server.is_alive():
        ca_server.join()


def start_envoy(tls=False):
    envoy_folder = WORKSPACE_PATH / 'envoy'
    with set_directory(envoy_folder):
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'sd_requirements.txt'
        ])
        params = [
            '-n', 'one',
            '-dh', 'localhost',
            '-dp', '50051',
            '--disable-tls',
            '-ec', 'envoy_config.yaml'
        ]
        if not tls:
            params.append('--disable-tls')
        envoy = subprocess.Popen([
            'fx', 'envoy', 'start',
            *params
        ])
        return envoy


if __name__ == '__main__':
    shutil.rmtree(CA_PATH, ignore_errors=True)
    install_ca()
    ca_server = start_ca_server()
    try:
        token = get_token()
        certify_director(token)
        director = start_director(tls=True)
        time.sleep(3)  # Wait for Director to start
        try:
            envoy = start_envoy(tls=False)
            time.sleep(3)  # Wait for envoy to start
            # Expectation is that secure server won't let insecure client connect
            while True:
                tmp = director.stderr.readline().decode('utf-8')
                if EXPECTED_ERROR in tmp:
                    print('Handshake failed for insecure connections as expected.')
                    break
        finally:
            director.kill()
    finally:
        stop_server(ca_server)
