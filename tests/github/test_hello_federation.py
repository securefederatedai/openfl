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


def create_collaborator(col, workspace_root, data_path):
    col_path = workspace_root / col
    shutil.rmtree(col_path, ignore_errors=True)
    col_path.mkdir()
    os.chdir(col_path)
    check_call(['fx', 'workspace', 'import', '--archive', workspace_root / archive_name])

    os.chdir(col_path / fed_workspace)
    check_call([
        'fx', 'collaborator', 'generate-cert-request', '-d', data_path, '-n', col, '--silent'
    ])

    os.chdir(workspace_root)
    request_pkg = col_path / fed_workspace / f'col_{col}_to_agg_cert_request.zip'
    check_call(['fx', 'collaborator', 'certify', '--request-pkg', f'{request_pkg}', '--silent'])

    os.chdir(col_path / fed_workspace)
    import_path = workspace_root / f'agg_to_col_{col}_signed_cert.zip'
    check_call(['fx', 'collaborator', 'certify', '--import', import_path])


def exec(command, directory):
    os.chdir(directory)
    check_call(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='keras_cnn_mnist')
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
    col1, col2 = args.col1, args.col2
    col1_data_path, col2_data_path = args.col1_data_path, args.col2_data_path
    shutil.rmtree(fed_workspace, ignore_errors=True)
    check_call(['fx', 'workspace', 'create', '--prefix', fed_workspace, '--template', template])
    os.chdir(fed_workspace)
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

    check_call(['fx', 'workspace', 'certify'])
    check_call(['fx', 'workspace', 'export'])
    check_call(['fx', 'aggregator', 'generate-cert-request', '--fqdn', fqdn])
    check_call(['fx', 'aggregator', 'certify', '--fqdn', fqdn, '--silent'])

    workspace_root = Path().resolve()
    create_collaborator(col1, workspace_root, col1_data_path)
    create_collaborator(col2, workspace_root, col2_data_path)
    time.sleep(10)

    with ProcessPoolExecutor() as executor:
        executor.submit(exec, ['fx', 'aggregator', 'start'], workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(exec, ['fx', 'collaborator', 'start', '-n', col1], dir1)

        dir2 = workspace_root / col2 / fed_workspace
        executor.submit(exec, ['fx', 'collaborator', 'start', '-n', col2], dir2)
    shutil.rmtree(workspace_root)
