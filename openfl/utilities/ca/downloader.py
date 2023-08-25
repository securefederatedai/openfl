# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import urllib.request
import shutil

from click import confirm

VERSION = '0.16.0'

ARCHITECTURE_ALIASES = {
    'x86_64': 'amd64',
    'armv6l': 'armv6',
    'armv7l': 'armv7',
    'aarch64': 'arm64'
}

FILE_EXTENSIONS = {
    'windows': 'zip',
    'linux': 'tar.gz'
}


def get_system_and_architecture():
    """Get system and architecture of machine."""
    uname_res = platform.uname()
    system = uname_res.system.lower()

    architecture = uname_res.machine.lower()
    architecture = ARCHITECTURE_ALIASES.get(architecture, architecture)

    return system, architecture


def download_step_bin(prefix='.', confirmation=True):
    """
    Download step binaries.

    Args:
        prefix: folder path to download
        confirmation: request user confirmation or not
    """
    system, arch = get_system_and_architecture()
    ext = FILE_EXTENSIONS[system]
    binary = f'step_{system}_{VERSION}_{arch}.{ext}'
    url = f'https://dl.step.sm/gh-release/cli/docs-cli-install/v{VERSION}/{binary}'
    _download(url, prefix, confirmation)


def download_step_ca_bin(prefix='.', confirmation=True):
    """
    Download step-ca binaries.

    Args:
        prefix: folder path to download
        confirmation: request user confirmation or not
    """
    system, arch = get_system_and_architecture()
    ext = FILE_EXTENSIONS[system]
    binary = f'step-ca_{system}_{VERSION}_{arch}.{ext}'
    url = f'https://dl.step.sm/gh-release/certificates/docs-ca-install/v{VERSION}/{binary}'
    _download(url, prefix, confirmation)


def _download(url, prefix, confirmation):
    if confirmation:
        confirm('CA binaries will be downloaded now', default=True, abort=True)
    name = url.split('/')[-1]
    urllib.request.urlretrieve(url, f'{prefix}/{name}')
    shutil.unpack_archive(f'{prefix}/{name}', f'{prefix}/step')
