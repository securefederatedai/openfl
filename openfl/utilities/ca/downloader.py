# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import platform
import shutil
import urllib.request

from click import confirm

VERSION = "0.16.0"

ARCHITECTURE_ALIASES = {
    "x86_64": "amd64",
    "armv6l": "armv6",
    "armv7l": "armv7",
    "aarch64": "arm64",
}

FILE_EXTENSIONS = {"windows": "zip", "linux": "tar.gz"}


def get_system_and_architecture():
    """Get the system and architecture of the machine.

    Returns:
        tuple: The system and architecture of the machine.
    """
    uname_res = platform.uname()
    system = uname_res.system.lower()

    architecture = uname_res.machine.lower()
    architecture = ARCHITECTURE_ALIASES.get(architecture, architecture)

    return system, architecture


def download_step_bin(prefix=".", confirmation=True):
    """
    Download step binaries.

    Args:
        prefix (str, optional): Folder path to download. Defaults to '.'.
        confirmation (bool, optional): Request user confirmation or not.
            Defaults to True.
    """
    system, arch = get_system_and_architecture()
    ext = FILE_EXTENSIONS[system]
    binary = f"step_{system}_{VERSION}_{arch}.{ext}"
    url = f"https://dl.step.sm/gh-release/cli/docs-cli-install/v{VERSION}/{binary}"
    _download(url, prefix, confirmation)


def download_step_ca_bin(prefix=".", confirmation=True):
    """
    Download step-ca binaries.

    Args:
        prefix (str, optional): Folder path to download. Defaults to '.'.
        confirmation (bool, optional): Request user confirmation or not.
            Defaults to True.
    """
    system, arch = get_system_and_architecture()
    ext = FILE_EXTENSIONS[system]
    binary = f"step-ca_{system}_{VERSION}_{arch}.{ext}"
    url = f"https://dl.step.sm/gh-release/certificates/docs-ca-install/v{VERSION}/{binary}"
    _download(url, prefix, confirmation)


def _download(url, prefix, confirmation):
    """Download a file from a URL.

    Args:
        url (str): URL of the file to download.
        prefix (str): Folder path to download.
        confirmation (bool): Request user confirmation or not.
    """
    if confirmation:
        confirm("CA binaries will be downloaded now", default=True, abort=True)
    name = url.split("/")[-1]
    # nosec: private function definition with static urls
    urllib.request.urlretrieve(url, f"{prefix}/{name}")  # nosec
    shutil.unpack_archive(f"{prefix}/{name}", f"{prefix}/step")
