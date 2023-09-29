# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CA module."""

import base64
import json
import os
import sys
import shutil
import signal
import subprocess  # nosec
import time
from logging import getLogger
from pathlib import Path
from subprocess import check_call  # nosec

from click import confirm

from openfl.utilities.ca.downloader import download_step_bin
from openfl.utilities.ca.downloader import download_step_ca_bin

logger = getLogger(__name__)

TOKEN_DELIMITER = '.'
CA_STEP_CONFIG_DIR = Path('step_config')
CA_PKI_DIR = Path('cert')
CA_PASSWORD_FILE = Path('pass_file')
CA_CONFIG_JSON = Path('config/ca.json')


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
        sys.exit(1)

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
                step_executable = 'step'
                if sys.platform == 'win32':
                    step_executable = 'step.exe'
                step = ca_path / 'step' / dir_ / 'bin' / step_executable
            if 'step-ca' in dir_:
                step_ca_executable = 'step-ca'
                if sys.platform == 'win32':
                    step_ca_executable = 'step-ca.exe'
                step_ca = ca_path / 'step' / dir_ / 'bin' / step_ca_executable
    return step, step_ca


def certify(name, cert_path: Path, token_with_cert, ca_path: Path):
    """Create an envoy workspace."""
    os.makedirs(cert_path, exist_ok=True)

    token, root_certificate = token_with_cert.split(TOKEN_DELIMITER)
    token = base64.b64decode(token).decode('utf-8')
    root_certificate = base64.b64decode(root_certificate)

    step_path, _ = get_ca_bin_paths(ca_path)
    if not step_path:
        download_step_bin(prefix=ca_path)
        step_path, _ = get_ca_bin_paths(ca_path)
    if not step_path:
        raise Exception('Step-CA is not installed!\nRun `fx pki install` first')

    with open(f'{cert_path}/root_ca.crt', mode='wb') as file:
        file.write(root_certificate)
    check_call(f'{step_path} ca certificate {name} {cert_path}/{name}.crt '
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
        download_step_bin(prefix=ca_path, confirmation=True)
        download_step_ca_bin(prefix=ca_path, confirmation=False)
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR
    if (not step_config_dir.exists()
            or confirm('CA exists, do you want to recreate it?', default=True)):
        _create_ca(ca_path, ca_url, password)
    _configure(step_config_dir)


def run_ca(step_ca, pass_file, ca_json):
    """Run CA server."""
    if _check_kill_process('step-ca', confirmation=True):
        logger.info('Up CA server')
        check_call(f'{step_ca} --password-file {pass_file} {ca_json}', shell=True)


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

    with open(f'{pki_dir}/pass_file', 'w', encoding='utf-8') as f:
        f.write(password)
    os.chmod(f'{pki_dir}/pass_file', 0o600)
    step_path, step_ca_path = get_ca_bin_paths(ca_path)
    if not (step_path and step_ca_path and step_path.exists() and step_ca_path.exists()):
        logger.error('Could not find step-ca binaries in the path specified')
        sys.exit(1)

    logger.info('Create CA Config')
    os.environ['STEPPATH'] = str(step_config_dir)
    shutil.rmtree(step_config_dir, ignore_errors=True)
    name = ca_url.split(':')[0]
    check_call(
        f'{step_path} ca init --name name --dns {name} '
        f'--address {ca_url}  --provisioner prov '
        f'--password-file {pki_dir}/pass_file',
        shell=True
    )

    check_call(f'{step_path} ca provisioner remove prov --all', shell=True)
    check_call(
        f'{step_path} crypto jwk create {step_config_dir}/certs/pub.json '
        f'{step_config_dir}/secrets/priv.json --password-file={pki_dir}/pass_file',
        shell=True
    )
    check_call(
        f'{step_path} ca provisioner add provisioner {step_config_dir}/certs/pub.json',
        shell=True
    )


def _configure(step_config_dir):
    conf_file = step_config_dir / CA_CONFIG_JSON
    with open(conf_file, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.setdefault('authority', {}).setdefault('claims', {})
        data['authority']['claims']['maxTLSCertDuration'] = f'{365 * 24}h'
        data['authority']['claims']['defaultTLSCertDuration'] = f'{365 * 24}h'
        data['authority']['claims']['maxUserSSHCertDuration'] = '24h'
        data['authority']['claims']['defaultUserSSHCertDuration'] = '24h'
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
