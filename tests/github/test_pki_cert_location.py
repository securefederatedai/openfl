# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
import argparse
from pathlib import Path
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor

from openfl.utilities.utils import rmtree
from tests.github.utils import create_collaborator, create_certified_workspace, certify_aggregator


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
    parser.add_argument('--agg-cert-path', default=Path.cwd())
    parser.add_argument('--col1-cert-path', default=Path.cwd())
    parser.add_argument('--col2-cert-path', default=Path.cwd())
    parser.add_argument('--save-model')

    origin_dir = Path.cwd().resolve()
    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    archive_name = f'{fed_workspace}.zip'
    fqdn = socket.getfqdn()
    template = args.template
    rounds_to_train = args.rounds_to_train
    col1, col2 = args.col1, args.col2
    col1_data_path, col2_data_path = args.col1_data_path, args.col2_data_path
    agg_cert_path = Path(args.agg_cert_path).resolve()
    col1_cert_path = Path(args.col1_cert_path).resolve()
    col2_cert_path = Path(args.col2_cert_path).resolve()
    save_model = args.save_model

    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.
    create_certified_workspace(fed_workspace, template, fqdn, rounds_to_train, agg_cert_path)
    certify_aggregator(fqdn, agg_cert_path)

    workspace_root = Path().resolve()  # Get the absolute directory path for the workspace

    # Create collaborator #1
    create_collaborator(col1, workspace_root, col1_data_path, archive_name,
                        fed_workspace, col1_cert_path, agg_cert_path)

    # Create collaborator #2
    create_collaborator(col2, workspace_root, col2_data_path, archive_name,
                        fed_workspace, col2_cert_path, agg_cert_path)

    # Run the federation
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(
            check_call, ['fx', 'aggregator', 'start', '-c', agg_cert_path],
            cwd=workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(
            check_call, ['fx', 'collaborator', 'start', '-n', col1, '-c', col1_cert_path],
            cwd=dir1)

        dir2 = workspace_root / col2 / fed_workspace
        executor.submit(
            check_call, ['fx', 'collaborator', 'start', '-n', col2, '-c', col2_cert_path],
            cwd=dir2)

    # Convert model to native format
    if save_model:
        check_call(
            ['fx', 'model', 'save', '-i', f'./save/{template}_last.pbuf', '-o', save_model],
            cwd=workspace_root)

    # Clear cert paths
    check_call(
        ['fx', 'aggregator', 'uninstall-cert', '-c', agg_cert_path],
        cwd=workspace_root)
    check_call(
        ['fx', 'collaborator', 'uninstall-cert', '-c', col1_cert_path],
        cwd=workspace_root / col1 / fed_workspace)
    check_call(
        ['fx', 'collaborator', 'uninstall-cert', '-c', col2_cert_path],
        cwd=workspace_root / col2 / fed_workspace)
    os.chdir(origin_dir)
    rmtree(workspace_root)
