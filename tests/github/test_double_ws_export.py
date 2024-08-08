# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
import argparse
from pathlib import Path
import shutil
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor
import psutil

from tests.github.utils import create_certified_workspace, certify_aggregator, create_collaborator
from openfl.utilities.utils import getfqdn_env

if __name__ == '__main__':
    # Test the pipeline
    parser = argparse.ArgumentParser()
    workspace_choice = []
    with os.scandir('openfl-workspace') as iterator:
        for entry in iterator:
            if entry.name not in ['__init__.py', 'workspace', 'default']:
                workspace_choice.append(entry.name)
    parser.add_argument('--template', default='keras_cnn_mnist', choices=workspace_choice)
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')
    parser.add_argument('--col1', default='one123dragons')
    parser.add_argument('--col2', default='beta34unicorns')
    parser.add_argument('--rounds-to-train')
    parser.add_argument('--col1-data-path', default='1')
    parser.add_argument('--col2-data-path', default='2')

    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    archive_name = f'{fed_workspace}.zip'
    fqdn = getfqdn_env()
    template = args.template
    rounds_to_train = args.rounds_to_train
    col1 = args.col1
    col1_data_path = args.col1_data_path

    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.
    create_certified_workspace(fed_workspace, template, fqdn, rounds_to_train)

    certify_aggregator(fqdn)

    workspace_root = Path().resolve()  # Get the absolute directory path for the workspace

    # Create collaborator #1
    create_collaborator(col1, workspace_root, col1_data_path, archive_name, fed_workspace)

    # Run the federation
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(check_call, ['fx', 'aggregator', 'start'], cwd=workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(check_call, ['fx', 'collaborator', 'start', '-n', col1], cwd=dir1)
    shutil.rmtree(dir1)
    for proc in psutil.process_iter():
        if 'fx' in proc.name():
            proc.kill()
    # Initialize FL plan
    check_call(['fx', 'plan', 'initialize', '-a', fqdn])
    # Create certificate authority for workspace
    check_call(['fx', 'workspace', 'certify'])

    # Export FL workspace
    check_call(['fx', 'workspace', 'export'])

    certify_aggregator(fqdn)

    # Create collaborator #1
    create_collaborator(col1, workspace_root, col1_data_path, archive_name, fed_workspace)
    # Run the federation
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(check_call, ['fx', 'aggregator', 'start'], cwd=workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(check_call, ['fx', 'collaborator', 'start', '-n', col1], cwd=dir1)
    shutil.rmtree(workspace_root)
