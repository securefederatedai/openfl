# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Workspace module."""

from pathlib import Path
from click import Choice, Path as ClickPath
from click import group, option, pass_context
from click import echo, confirm
from subprocess import check_call
from sys import executable
from shutil import copyfile, ignore_patterns
import os, signal

from openfl.interface.cli_helper import copytree, print_tree

pki_dir = './cert'
step_config_dir = './step_config/'
step = './step/step_0.15.16/bin/step'
step_ca = './step/step-ca_0.15.15/bin/step-ca'


@group()
@pass_context
def ca(context):
    """Manage Federated Learning Workspaces."""
    context.obj['group'] = 'ca'


def check_kill_process(pstring):
    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)

def create_dirs():
    """Create workspace directories."""
    echo('Creating Workspace Directories')
    prefix = Path('.')
    (prefix / 'cert').mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / step_config_dir).mkdir(parents=True, exist_ok=True)  # certifications

#./step ca token localhost --ca-url https://nnlicv674.inn.intel.com:4343 --root step-config/certs/root_ca.crt
@ca.command(name='get_token')
@option('--name',
        required=True,
        help='dns naame or collaborator name')
def get_token_(name):
    """Create certificate authority for federation."""
    get_token(name)


def get_token(name):
    """Create certificate authority for federation."""
    import subprocess
    os.environ["STEPPATH"] = step_config_dir
    try:
        token = subprocess.check_output(f'./{step} ca token {name} --key ./step_config/secrets/priv.json ' +
              f'--password-file {pki_dir}/pass --ca-url https://nnlicv674.inn.intel.com:4343 ',shell=True)
    except subprocess.CalledProcessError as exc:                                                                                                   
        print("error code", exc.returncode, exc.output)
        return
    
    if token[-1:] == b'\n':
        token = token[:-1]
    length = len(token)
    assert(length < 10000)
    length = str(10000 + length)[-4:]
    with open(step_config_dir + '/certs/root_ca.crt',mode='rb') as file:
        root_ca = file.read()
    import base64

    message = root_ca 
    base64_bytes = base64.b64encode(message)
    base64_message = base64_bytes.decode('ascii')
    print('Token:')
    print(length, token.decode('ascii'),base64_message, sep='')



@ca.command(name='write_token')
@option('--name',
        required=True,
        help='dns naame or collaborator name')
def write_token_(name):
    """Create certificate authority for federation."""
    write_token(name)


def write_token(name):
    print(name)
    with open(step_config_dir + '/cert/root_ca.crt',mode='w') as file:
        file.write(name)

def download_step(url, grep_name, prefix='./'):
    import requests 
    import shutil
    result = requests.get(url)
    assets = result.json()['assets']
    urls = []
    for a in assets:
        if grep_name in a['name'] and 'amd' in a['name'] :
            urls.append(a['browser_download_url'])
    url = urls[-1]
    url.replace('https','http')
    name = url.split('/')[-1]
    import urllib.request
    print('Downloading:', name)
    urllib.request.urlretrieve(url, f'{prefix}/{name}')
    import shutil
    shutil.unpack_archive(f'{prefix}/{name}', f'{prefix}/step')

@ca.command(name='init')
@option('--password', required=True,
        help='Password to encrypted keys')
def init(password):
    """Create certificate authority for federation."""
    from openfl.cryptography.ca import generate_root_cert, generate_signing_csr, sign_certificate
    from cryptography.hazmat.primitives import serialization
    check_kill_process('step-ca')
    create_dirs() 
    with open(f'{pki_dir}/pass', 'w') as f:
        f.write(password)

    echo('Setting Up Certificate Authority...\n')

    import shutil

    # root_crt_path = 'ca/root-ca.crt'
    # root_key_path = 'ca/root-ca/private/root-ca.key'

    #root_private_key, root_cert = generate_root_cert()
    url = 'http://api.github.com/repos/smallstep/certificates/releases/latest'
    download_step(url, 'step-ca_linux')
    url = 'http://api.github.com/repos/smallstep/cli/releases/latest'
    download_step(url,'step_linux')
    echo('Create CA Config')
    os.environ["STEPPATH"] = step_config_dir
    shutil.rmtree(step_config_dir, ignore_errors=True)
    os.system(f'./{step} ca init --name name --dns nnlicv674.inn.intel.com ' +
                    f'--address nnlicv674.inn.intel.com:4343  --provisioner prov ' +
                    f'--password-file {pki_dir}/pass')

    os.system(f'./{step} ca provisioner remove prov --all')
    os.system(f'./{step} crypto jwk create ./step_config/certs/pub.json ' + 
              f'./step_config/secrets/priv.json --password-file={pki_dir}/pass')
    os.system(f'./{step} ca provisioner add provisioner ./step_config/certs/pub.json')
    echo('Up CA server')
    os.system(f'{step_ca} --password-file {pki_dir}/pass $(./{step} path)/config/ca.json  &')
    echo('\nDone.')


