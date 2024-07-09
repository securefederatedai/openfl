# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    """Get system and architecture of machine."""
    uname_res = platform.uname()
    system = uname_res.system.lower()

    architecture = uname_res.machine.lower()
    architecture = ARCHITECTURE_ALIASES.get(architecture, architecture)

    return system, architecture


def download_step_bin(prefix=".", confirmation=True):
    """
    Download step binaries.

    Args:
        prefix: folder path to download
        confirmation: request user confirmation or not
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
        prefix: folder path to download
        confirmation: request user confirmation or not
    """
    system, arch = get_system_and_architecture()
    ext = FILE_EXTENSIONS[system]
    binary = f"step-ca_{system}_{VERSION}_{arch}.{ext}"
    url = f"https://dl.step.sm/gh-release/certificates/docs-ca-install/v{VERSION}/{binary}"
    _download(url, prefix, confirmation)


def _download(url, prefix, confirmation):
    if confirmation:
        confirm("CA binaries will be downloaded now", default=True, abort=True)
    name = url.split("/")[-1]
    # nosec: private function definition with static urls
    urllib.request.urlretrieve(url, f"{prefix}/{name}")  # nosec
    shutil.unpack_archive(f"{prefix}/{name}", f"{prefix}/step")
