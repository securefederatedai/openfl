# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregator module."""

import base64
import json
import os
import platform
import shutil
import signal
import subprocess
import time
import urllib.request
from logging import getLogger
from pathlib import Path
from subprocess import call

import requests
from click import confirm

logger = getLogger(__name__)

TOKEN_DELIMITER = '.'
CA_STEP_CONFIG_DIR = Path('step_config')
CA_PKI_DIR = Path('cert')
CA_PASSWORD_FILE = Path('pass_file')
CA_CONFIG_JSON = Path('config/ca.json')


def get_system_and_architecture():
    """Get system and architecture of machine."""
    uname_res = platform.uname()
    system = uname_res.system.lower()

    architecture_aliases = {
        'x86_64': 'amd64',
        'armv6l': 'armv6',
        'armv7l': 'armv7',
        'aarch64': 'arm64'
    }
    architecture = uname_res.machine.lower()
    for alias in architecture_aliases:
        if architecture == alias:
            architecture = architecture_aliases[alias]
            break

    return system, architecture


def download_step_bin(url, grep_name, architecture, prefix='.', confirmation=True):
    """
    Donwload step binaries from github.

    Args:
        url: address of latest release
        grep_name: name to grep over github assets
        architecture: architecture type to grep
        prefix: folder path to download
        confirmation: request user confirmation or not
    """
    if confirmation:
        confirm('CA binaries from github will be downloaded now', default=True, abort=True)
    result = requests.get(url)
    if result.status_code != 200:
        logger.warning('Can\'t download binaries from github. Please try lately.')
        return

    assets = result.json().get('assets', [])
    archive_urls = [
        a['browser_download_url']
        for a in assets
        if (grep_name in a['name'] and architecture in a['name']
            and 'application/gzip' in a['content_type'])
    ]
    if len(archive_urls) == 0:
        raise Exception('Applicable CA binaries from github were not found '
                        f'(name: {grep_name}, architecture: {architecture})')
    archive_url = archive_urls[-1]
    archive_url = archive_url.replace('https', 'http')
    name = archive_url.split('/')[-1]
    logger.info(f'Downloading {name}')
    urllib.request.urlretrieve(archive_url, f'{prefix}/{name}')
    shutil.unpack_archive(f'{prefix}/{name}', f'{prefix}/step')


def get_token(name, ca_url, ca_path='.'):
    """
    Create authentication token.

    Args:
        name: common name for following certificate
                    (aggregator fqdn or collaborator name)
        ca_url: full url of CA server
        ca_path: path to ca folder
    """
    ca_path = Path(ca_path)
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR
    pki_dir = ca_path / CA_PKI_DIR
    step_path, _ = get_ca_bin_paths(ca_path)
    if not step_path:
        raise Exception('Step-CA is not installed!\nRun `fx pki install` first')

    priv_json = step_config_dir / 'secrets' / 'priv.json'
    pass_file = pki_dir / CA_PASSWORD_FILE
    root_crt = step_config_dir / 'certs' / 'root_ca.crt'
    try:
        token = subprocess.check_output(
            f'{step_path} ca token {name} '
            f'--key {priv_json} --root {root_crt} '
            f'--password-file {pass_file} 'f'--ca-url {ca_url}', shell=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f'Error code {exc.returncode}: {exc.output}')
        return

    token = token.strip()
    token_b64 = base64.b64encode(token)

    with open(root_crt, mode='rb') as file:
        root_certificate_b = file.read()
    root_ca_b64 = base64.b64encode(root_certificate_b)

    return TOKEN_DELIMITER.join([
        token_b64.decode('utf-8'),
        root_ca_b64.decode('utf-8'),
    ])


def get_ca_bin_paths(ca_path):
    """Get paths of step binaries."""
    ca_path = Path(ca_path)
    step = None
    step_ca = None
    if (ca_path / 'step').exists():
        dirs = os.listdir(ca_path / 'step')
        for dir_ in dirs:
            if 'step_' in dir_:
                step = ca_path / 'step' / dir_ / 'bin' / 'step'
            if 'step-ca' in dir_:
                step_ca = ca_path / 'step' / dir_ / 'bin' / 'step-ca'
    return step, step_ca


def certify(name, cert_path: Path, token_with_cert, ca_path: Path):
    """Create an envoy workspace."""
    os.makedirs(cert_path, exist_ok=True)

    token, root_certificate = token_with_cert.split(TOKEN_DELIMITER)
    token = base64.b64decode(token).decode('utf-8')
    root_certificate = base64.b64decode(root_certificate)

    step_path, _ = get_ca_bin_paths(ca_path)
    if not step_path:
        url = 'https://api.github.com/repos/smallstep/cli/releases/latest'
        system, arch = get_system_and_architecture()
        download_step_bin(url, f'step_{system}', arch, prefix=ca_path)
        step_path, _ = get_ca_bin_paths(ca_path)
    if not step_path:
        raise Exception('Step-CA is not installed!\nRun `fx pki install` first')

    with open(f'{cert_path}/root_ca.crt', mode='wb') as file:
        file.write(root_certificate)
    call(f'{step_path} ca certificate {name} {cert_path}/{name}.crt '
         f'{cert_path}/{name}.key --kty EC --curve P-384 -f --token {token}', shell=True)


def remove_ca(ca_path):
    """Kill step-ca process and rm ca directory."""
    _check_kill_process('step-ca')
    shutil.rmtree(ca_path, ignore_errors=True)


def install(ca_path, ca_url, password):
    """
    Create certificate authority for federation.

    Args:
        ca_path: path to ca directory
        ca_url: url for ca server like: 'host:port'
        password: Simple password for encrypting root private keys

    """
    logger.info('Creating CA')

    ca_path = Path(ca_path)
    ca_path.mkdir(parents=True, exist_ok=True)
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR
    os.environ['STEPPATH'] = str(step_config_dir)
    step_path, step_ca_path = get_ca_bin_paths(ca_path)

    if not (step_path and step_ca_path and step_path.exists() and step_ca_path.exists()):
        confirm('CA binaries from github will be downloaded now', default=True, abort=True)
        system, arch = get_system_and_architecture()
        url = 'https://api.github.com/repos/smallstep/certificates/releases/latest'
        download_step_bin(url, f'step-ca_{system}', arch, prefix=ca_path, confirmation=False)
        url = 'https://api.github.com/repos/smallstep/cli/releases/latest'
        download_step_bin(url, f'step_{system}', arch, prefix=ca_path, confirmation=False)
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR
    if (not step_config_dir.exists()
            or confirm('CA exists, do you want to recreate it?', default=True)):
        _create_ca(ca_path, ca_url, password)
    _configure(step_config_dir)


def run_ca(step_ca, pass_file, ca_json):
    """Run CA server."""
    if _check_kill_process('step-ca', confirmation=True):
        logger.info('Up CA server')
        call(f'{step_ca} --password-file {pass_file} {ca_json}', shell=True)


def _check_kill_process(pstring, confirmation=False):
    """Kill process by name."""
    pids = []
    proc = subprocess.Popen(f'ps ax | grep {pstring} | grep -v grep',
                            shell=True, stdout=subprocess.PIPE)
    text = proc.communicate()[0].decode('utf-8')

    for line in text.splitlines():
        fields = line.split()
        pids.append(fields[0])

    if len(pids):
        if confirmation and not confirm('CA server is already running. Stop him?', default=True):
            return False
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
        time.sleep(2)
    return True


def _create_ca(ca_path: Path, ca_url: str, password: str):
    """Create a ca workspace."""
    import os
    pki_dir = ca_path / CA_PKI_DIR
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR

    pki_dir.mkdir(parents=True, exist_ok=True)
    step_config_dir.mkdir(parents=True, exist_ok=True)

    with open(f'{pki_dir}/pass_file', 'w') as f:
        f.write(password)
    os.chmod(f'{pki_dir}/pass_file', 0o600)
    step_path, step_ca_path = get_ca_bin_paths(ca_path)
    assert (step_path and step_ca_path and step_path.exists() and step_ca_path.exists())

    logger.info('Create CA Config')
    os.environ['STEPPATH'] = str(step_config_dir)
    shutil.rmtree(step_config_dir, ignore_errors=True)
    name = ca_url.split(':')[0]
    call(f'{step_path} ca init --name name --dns {name} '
         f'--address {ca_url}  --provisioner prov '
         f'--password-file {pki_dir}/pass_file', shell=True)

    call(f'{step_path} ca provisioner remove prov --all', shell=True)
    call(f'{step_path} crypto jwk create {step_config_dir}/certs/pub.json '
         f'{step_config_dir}/secrets/priv.json --password-file={pki_dir}/pass_file', shell=True)
    call(
        f'{step_path} ca provisioner add provisioner {step_config_dir}/certs/pub.json',
        shell=True
    )


def _configure(step_config_dir):
    conf_file = step_config_dir / CA_CONFIG_JSON
    with open(conf_file, 'r+') as f:
        data = json.load(f)
        data.setdefault('authority', {}).setdefault('claims', {})
        data['authority']['claims']['maxTLSCertDuration'] = f'{365 * 24}h'
        data['authority']['claims']['defaultTLSCertDuration'] = f'{365 * 24}h'
        data['authority']['claims']['maxUserSSHCertDuration'] = '24h'
        data['authority']['claims']['defaultUserSSHCertDuration'] = '24h'
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
