# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Generic check functions."""

import os


def get_credentials(folder_path):
    """Get credentials from folder by template.

    This function retrieves the root certificate, key, and certificate from
    the specified folder.
    The files are identified by their extensions: '.key' for the key, '.crt'
    for the certificate, and 'root_ca' for the root certificate.

    Args:
        folder_path (str): The path to the folder containing the credentials.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: The paths to the
            root certificate, key, and certificate.
            If a file is not found, its corresponding value is None.
    """
    root_ca, key, cert = None, None, None
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if ".key" in f:
                key = folder_path + os.sep + f
            if ".crt" in f and "root_ca" not in f:
                cert = folder_path + os.sep + f
            if "root_ca" in f:
                root_ca = folder_path + os.sep + f
    return root_ca, key, cert
