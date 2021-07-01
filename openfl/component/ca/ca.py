# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregator module."""


import requests
import urllib.request
import shutil
import os 
from pathlib import Path
from logging import getLogger

from openfl.databases import TensorDB
from openfl.pipelines import NoCompressionPipeline
from openfl.pipelines import TensorCodec
from openfl.protocols import ModelProto
from openfl.protocols import utils
from openfl.utilities import TaskResultKey
from openfl.utilities import TensorKey
from openfl.utilities.logs import write_metric

logger = getLogger(__name__)


def download_step_bin(url, grep_name, architecture, prefix='./'):
    """
    Donwload step binaries from github.

    Args:
        url: address of latest release
        grep_name: name to grep over github assets
        architecture: architecture type to grep
        prefix: folder path to download
    """

    result = requests.get(url)
    assets = result.json()['assets']
    urls = []
    for a in assets:
        if grep_name in a['name'] and architecture in a['name']:  # 'amd'
            urls.append(a['browser_download_url'])
    url = urls[-1]
    url = url.replace('https', 'http')
    name = url.split('/')[-1]
    print('Downloading:', name)
    urllib.request.urlretrieve(url, f'{prefix}/{name}')
    shutil.unpack_archive(f'{prefix}/{name}', f'{prefix}/step')

def get_token(name, ca_url):
    """
    Create authentication token.

    Args:
        name: common name for following certificate
                    (aggregator fqdn or collaborator name)
        ca_url: full url of CA server
    """
    import subprocess
    import os
    import base64 
    step = './step/step_0.15.16/bin/step'
    step_config_dir = './step_config/'
    pki_dir = './cert'

    os.environ["STEPPATH"] = step_config_dir
    try:
        token = subprocess.check_output(f'./{step} ca token {name} 'f'--key {step_config_dir}/secrets/priv.json 'f'--password-file {pki_dir}/pass_file 'f'--ca-url {ca_url} ', shell=True)
    except subprocess.CalledProcessError as exc:
        print(f'Error code {exc.returncode}: {exc.output}')
        return

    if token[-1:] == b'\n':
        token = token[:-1]
    length = len(token)
    assert(length < 10000)
    length = str(10000 + length)[-4:]
    with open(step_config_dir + '/certs/root_ca.crt', mode='rb') as file:
        root_ca = file.read()

    base64_bytes = base64.b64encode(root_ca)
    base64_message = base64_bytes.decode('ascii')
    return str(length) + token.decode('ascii') + base64_message
    # print('Token:')
    # print(length, token.decode('ascii'),, sep='')

def get_bin_names(ca_path):
    import os
    step = None
    step_ca = None
    if os.path.exists(f'{ca_path}/step'): 
        dirs = os.listdir(f'{ca_path}/step')
        for dir_ in dirs:
            if 'step_' in dir_:
                step = f'{ca_path}/step/{dir_}/bin/step'
            if 'step-ca' in dir_:
                step_ca = f'{ca_path}/step/{dir_}/bin/step-ca'
    return step, step_ca


def certify(name, prefix_name,cert_path, token_with_cert):
    """Create a collaborator manager workspace."""
    import os
    import base64
    length = int(token_with_cert[:4])
    token = token_with_cert[4:length + 4]
    root_ca = token_with_cert[length + 4:]
    message_bytes = base64.b64decode(root_ca)

    step, _ = get_bin_names('.')
    if not step:
        url = 'http://api.github.com/repos/smallstep/cli/releases/latest'
        download_step_bin(url, 'step_linux', 'amd', prefix='./')
    step, _ = get_bin_names('.')
    assert(step)
    pki_dir = './cert'

    # write ca cert to file
    with open(f'{pki_dir}/root_ca.crt', mode='wb') as file:
        file.write(message_bytes)
    # request to ca server
    os.system(f'./{step} ca certificate {name} {cert_path}/{prefix_name}_{name}.crt {cert_path}/{prefix_name}_{name}.key -f --token {token}')

def run():
    """
    Create certificate authority for federation.

    Args:
        password: Simple password for encrypting root private keys
        ca_url: url for ca server like: 'host:port'

    """
    import os
    import threading

    check_kill_process('step-ca')
    print('Up CA server')

    step_config_dir = './step_config/'
    os.environ["STEPPATH"] = step_config_dir
    step, step_ca = get_bin_names('.')
    assert(step and step_ca and os.path.exists(step) and os.path.exists(step_ca))

    ca_thread = threading.Thread(
        target=lambda: os.system(
            f'{step_ca} --password-file ./cert/pass_file ./step_config/config/ca.json'
        ),
    )
    try:
        ca_thread.start()
    except Exception as exc:
        logger.error(f'Failed to up ca server: {exc}')

    print('\nDone.')

def check_kill_process(pstring):
    """Kill process by name."""
    import signal

    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)

def create(ca_path, ca_url, password):
    """Create a ca workspace."""
    import os

    step = None
    step_ca = None
    pki_dir = f'{ca_path}/cert'
    step_config_dir = f'{ca_path}/step_config/'

    prefix = Path('.')
    (prefix / pki_dir).mkdir(parents=True, exist_ok=True)
    (prefix / step_config_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{pki_dir}/pass_file', 'w') as f:
        f.write(password)
    step, step_ca = get_bin_names(ca_path)
    if not (step and step_ca):
        url = 'http://api.github.com/repos/smallstep/certificates/releases/latest'
        download_step_bin(url, 'step-ca_linux', 'amd', prefix=ca_path)
        url = 'http://api.github.com/repos/smallstep/cli/releases/latest'
        download_step_bin(url, 'step_linux', 'amd', prefix=ca_path)
    step, step_ca = get_bin_names(ca_path)
    assert(step and step_ca and os.path.exists(step) and os.path.exists(step_ca))

    print('Create CA Config')
    os.environ["STEPPATH"] = step_config_dir
    shutil.rmtree(step_config_dir, ignore_errors=True)
    name = ca_url.split(':')[0]
    os.system(f'./{step} ca init --name name --dns {name} '
            + f'--address {ca_url}  --provisioner prov '
            + f'--password-file {pki_dir}/pass_file')

    os.system(f'./{step} ca provisioner remove prov --all')
    os.system(f'./{step} crypto jwk create {step_config_dir}/certs/pub.json '
            + f'{step_config_dir}/secrets/priv.json --password-file={pki_dir}/pass_file')
    os.system(f'./{step} ca provisioner add provisioner {step_config_dir}/certs/pub.json')

