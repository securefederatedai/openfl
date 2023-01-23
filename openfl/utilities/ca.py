# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generic check functions."""
import os


def get_credentials(folder_path):
    """Get credentials from folder by template."""
    root_ca, key, cert = None, None, None
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if '.key' in f:
                key = folder_path + os.sep + f
            if '.crt' in f and 'root_ca' not in f:
                cert = folder_path + os.sep + f
            if 'root_ca' in f:
                root_ca = folder_path + os.sep + f
    return root_ca, key, cert
