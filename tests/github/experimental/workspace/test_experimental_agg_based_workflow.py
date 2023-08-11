# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
import argparse
from pathlib import Path
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor
from sys import executable
import shutil
from openfl.utilities.utils import rmtree
from tests.github.experimental.workspace.utils import create_collaborator
from tests.github.experimental.workspace.utils import create_certified_workspace
from tests.github.experimental.workspace.utils import certify_aggregator


if __name__ == '__main__':
    # Test the pipeline
    parser = argparse.ArgumentParser()
    workspace_choice = []
    with os.scandir('tests/github/experimental/workspace') as iterator:
        for entry in iterator:
            if entry.name not in ['__init__.py', 'workspace', 'default']:
                workspace_choice.append(entry.name)
    parser.add_argument('--template', default='testcase_include_exclude', choices=workspace_choice)
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')
    parser.add_argument('--col', action='append', default=[])
    parser.add_argument('--rounds-to-train')

    origin_dir = Path.cwd().resolve()
    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    archive_name = f'{fed_workspace}.zip'
    fqdn = socket.getfqdn()
    template = args.template
    rounds_to_train = args.rounds_to_train
    collaborators = args.col
    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.

    source_directory = origin_dir / 'tests'/'github'/'experimental'/'workspace' / template
    destination_directory = origin_dir / 'openfl-workspace' / 'experimental' / template
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)

    # Copy template to the destination directory
    shutil.copytree(src=source_directory, dst=destination_directory)

    check_call([executable, '-m', 'pip', 'install', '.'])

    # Activate experimental
    check_call(['fx', 'experimental', 'activate'])

    create_certified_workspace(fed_workspace, template, fqdn, rounds_to_train)
    certify_aggregator(fqdn)

    # Get the absolute directory path for the workspace
    workspace_root = Path().resolve()

    # Create Collaborators
    for collab in collaborators:
        create_collaborator(
            collab, workspace_root, archive_name, fed_workspace
        )

    # Run the federation
    with ProcessPoolExecutor(max_workers=len(collaborators) + 1) as executor:
        executor.submit(
            check_call, ['fx', 'aggregator', 'start'], cwd=workspace_root
        )
        time.sleep(5)

        for collab in collaborators:
            col_dir = workspace_root / collab / fed_workspace
            executor.submit(
                check_call, ['fx', 'collaborator', 'start', '-n', collab],
                cwd=col_dir
            )

    os.chdir(origin_dir)
    rmtree(workspace_root)

    # Remove template to the destination directory
    shutil.rmtree(destination_directory)

    # Deactivate experimental
    check_call(['fx', 'experimental', 'deactivate'])
