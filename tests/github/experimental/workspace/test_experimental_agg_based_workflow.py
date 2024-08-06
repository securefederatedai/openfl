# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
import argparse
from pathlib import Path
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor
from openfl.utilities.utils import rmtree
from tests.github.experimental.workspace.utils import create_collaborator
from tests.github.experimental.workspace.utils import create_certified_workspace
from tests.github.experimental.workspace.utils import certify_aggregator
from openfl.utilities.utils import getfqdn_env

if __name__ == '__main__':
    # Test the pipeline
    parser = argparse.ArgumentParser()
    workspace_choice = []
    with os.scandir('tests/github/experimental/workspace') as iterator:
        for entry in iterator:
            if entry.name not in ['__init__.py', 'workspace', 'default']:
                workspace_choice.append(entry.name)
    parser.add_argument('--custom_template')
    parser.add_argument('--template')
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')
    parser.add_argument('--col', action='append', default=[])
    parser.add_argument('--rounds-to-train')

    origin_dir = Path.cwd().resolve()
    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    archive_name = f'{fed_workspace}.zip'
    fqdn = getfqdn_env()
    template = args.template
    custom_template = args.custom_template
    rounds_to_train = args.rounds_to_train
    collaborators = args.col
    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.

    # Activate experimental
    check_call(['fx', 'experimental', 'activate'])

    create_certified_workspace(
        fed_workspace, custom_template, template, fqdn, rounds_to_train
    )
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

    # Deactivate experimental
    check_call(['fx', 'experimental', 'deactivate'])
