# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
import argparse
from pathlib import Path
import re
import shutil
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor
import psutil


def create_collaborator(col, workspace_root, data_path):
    # Copy workspace to collaborator directories (these can be on different machines)
    col_path = workspace_root / col
    shutil.rmtree(col_path, ignore_errors=True)  # Remove any existing directory
    col_path.mkdir()  # Create a new directory for the collaborator

    # Import the workspace to this collaborator
    check_call(
        ['fx', 'workspace', 'import', '--archive', workspace_root / archive_name],
        cwd=col_path
    )

    # Create collaborator certificate request
    # Remove '--silent' if you run this manually
    check_call(
        ['fx', 'collaborator', 'generate-cert-request', '-d', data_path, '-n', col, '--silent'],
        cwd=col_path / fed_workspace
    )

    # Sign collaborator certificate
    # Remove '--silent' if you run this manually
    request_pkg = col_path / fed_workspace / f'col_{col}_to_agg_cert_request.zip'
    check_call(
        ['fx', 'collaborator', 'certify', '--request-pkg', str(request_pkg), '--silent'],
        cwd=workspace_root)

    # Import the signed certificate from the aggregator
    import_path = workspace_root / f'agg_to_col_{col}_signed_cert.zip'
    check_call(
        ['fx', 'collaborator', 'certify', '--import', import_path],
        cwd=col_path / fed_workspace
    )


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
    fqdn = socket.getfqdn()
    template = args.template
    rounds_to_train = args.rounds_to_train
    col1 = args.col1
    col1_data_path = args.col1_data_path

    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.
    shutil.rmtree(fed_workspace, ignore_errors=True)
    check_call(['fx', 'workspace', 'create', '--prefix', fed_workspace, '--template', template])
    os.chdir(fed_workspace)

    # Initialize FL plan
    check_call(['fx', 'plan', 'initialize', '-a', fqdn])
    try:
        rounds_to_train = int(rounds_to_train)
        with open("plan/plan.yaml", "r") as sources:
            lines = sources.readlines()
        with open("plan/plan.yaml", "w") as sources:
            for line in lines:
                sources.write(
                    re.sub(r'rounds_to_train.*', f'rounds_to_train: {rounds_to_train}', line)
                )
    except (ValueError, TypeError):
        pass
    # Create certificate authority for workspace
    check_call(['fx', 'workspace', 'certify'])

    # Export FL workspace
    check_call(['fx', 'workspace', 'export'])

    # Create aggregator certificate
    check_call(['fx', 'aggregator', 'generate-cert-request', '--fqdn', fqdn])

    # Sign aggregator certificate
    check_call(['fx', 'aggregator', 'certify', '--fqdn', fqdn, '--silent'])

    workspace_root = Path().resolve()  # Get the absolute directory path for the workspace

    # Create collaborator #1
    create_collaborator(col1, workspace_root, col1_data_path)

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

    # Create aggregator certificate
    check_call(['fx', 'aggregator', 'generate-cert-request', '--fqdn', fqdn])

    # Sign aggregator certificate
    check_call(['fx', 'aggregator', 'certify', '--fqdn', fqdn, '--silent'])

    # Create collaborator #1
    create_collaborator(col1, workspace_root, col1_data_path)
    # Run the federation
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(check_call, ['fx', 'aggregator', 'start'], cwd=workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(check_call, ['fx', 'collaborator', 'start', '-n', col1], cwd=dir1)
    shutil.rmtree(workspace_root)
