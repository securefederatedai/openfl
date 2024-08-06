# Copyright (C) 2020-2023 Intel Corporation
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

from tests.github.utils import create_collaborator, certify_aggregator
from openfl.utilities.utils import getfqdn_env

def exec(command, directory):
    os.chdir(directory)
    check_call(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='keras_cnn_mnist')
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')
    parser.add_argument('--col1', default='one')
    parser.add_argument('--col2', default='two')
    parser.add_argument('--rounds-to-train')
    parser.add_argument('--col1-data-path', default='data/one')
    parser.add_argument('--col2-data-path', default='data/two')
    parser.add_argument('--gandlf_config', default=None)
    parser.add_argument('--ujjwal', action='store_true')

    origin_dir = Path().resolve()
    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    archive_name = f'{fed_workspace}.zip'
    fqdn = getfqdn_env()
    template = args.template
    rounds_to_train = args.rounds_to_train
    col1, col2 = args.col1, args.col2
    col1_data_path, col2_data_path = args.col1_data_path, args.col2_data_path
    shutil.rmtree(fed_workspace, ignore_errors=True)
    check_call(['fx', 'workspace', 'create', '--prefix', fed_workspace, '--template', template])
    os.chdir(fed_workspace)
    Path(Path.cwd().resolve() / 'data' / col1).mkdir(exist_ok=True)
    with os.scandir(origin_dir) as iterator:
        for entry in iterator:
            print(entry)
            if re.match(r'.*\.csv$', entry.name):
                shutil.copy(entry.path, Path.cwd().resolve() / 'data' / col1)
    # Initialize FL plan
    if args.gandlf_config:
        check_call(['fx', 'plan', 'initialize', '-a', fqdn,
                    '--gandlf_config', str(args.gandlf_config)])
    else:
        check_call(['fx', 'plan', 'initialize', '-a', fqdn])
    plan_path = Path('plan/plan.yaml')
    try:
        rounds_to_train = int(rounds_to_train)
        with open(plan_path, "r", encoding='utf-8') as sources:
            lines = sources.readlines()
        with open(plan_path, "w", encoding='utf-8') as sources:
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

    certify_aggregator(fqdn)
    if not Path('train.csv').exists():
        with os.scandir('..') as iterator:
            for entry in iterator:
                if re.match('train.csv', entry.name):
                    shutil.copy(entry.path, '.')
                if re.match('valid.csv', entry.name):
                    shutil.copy(entry.path, '.')

    workspace_root = Path().resolve()
    create_collaborator(col1, workspace_root, col1_data_path, archive_name, fed_workspace)
    create_collaborator(col2, workspace_root, col2_data_path, archive_name, fed_workspace)

    Path(workspace_root / col1 / fed_workspace / 'data' / col1).mkdir(exist_ok=True)
    Path(workspace_root / col2 / fed_workspace / 'data' / col2).mkdir(exist_ok=True)

    if args.ujjwal:
        with os.scandir('/media/ujjwal/SSD4TB/sbutil/DatasetForTraining_Horizontal/') as iterator:
            for entry in iterator:
                if re.match(r'^Site1_.*\.csv$', entry.name):
                    shutil.copy(entry.path, workspace_root / col1)
                if re.match(r'^Site2_.*\.csv$', entry.name):
                    shutil.copy(entry.path, workspace_root / col2)
    else:
        with os.scandir(workspace_root) as iterator:
            for entry in iterator:
                if re.match('train.csv', entry.name):
                    shutil.copy(entry.path, workspace_root / col1 / fed_workspace / 'data' / col1)
                    shutil.copy(entry.path, workspace_root / col2 / fed_workspace / 'data' / col2)
                if re.match('valid.csv', entry.name):
                    shutil.copy(entry.path, workspace_root / col1 / fed_workspace / 'data' / col1)
                    shutil.copy(entry.path, workspace_root / col2 / fed_workspace / 'data' / col2)

    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(exec, ['fx', 'aggregator', 'start'], workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(exec, ['fx', 'collaborator', 'start', '-n', col1], dir1)

        dir2 = workspace_root / col2 / fed_workspace
        executor.submit(exec, ['fx', 'collaborator', 'start', '-n', col2], dir2)
    shutil.rmtree(workspace_root)
