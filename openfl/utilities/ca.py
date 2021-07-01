# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CA utilities."""

import requests
import urllib.request
import shutil


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