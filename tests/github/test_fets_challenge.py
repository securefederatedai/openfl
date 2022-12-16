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
from utils import create_collaborator, create_certified_workspace, certify_aggregator


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
    parser.add_argument('--ujjwal', action='store_true')

    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    archive_name = f'{fed_workspace}.zip'
    fqdn = socket.getfqdn()
    template = args.template
    rounds_to_train = args.rounds_to_train
    col1, col2 = args.col1, args.col2
    col1_data_path, col2_data_path = args.col1_data_path, args.col2_data_path
    create_certified_workspace(fed_workspace, template, fqdn, rounds_to_train)
    certify_aggregator(fqdn)
    if not Path('seg_test_train.csv').exists():
        with os.scandir('..') as iterator:
            for entry in iterator:
                if re.match(r'^seg_test.*\.csv$', entry.name):
                    shutil.copy(entry.path, '.')
    workspace_root = Path().resolve()
    create_collaborator(col1, workspace_root, col1_data_path, archive_name, fed_workspace)
    create_collaborator(col2, workspace_root, col2_data_path, archive_name, fed_workspace)

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
                if re.match(r'^seg_test.*\.csv$', entry.name):
                    shutil.copy(entry.path, workspace_root / col1 / fed_workspace)
                    shutil.copy(entry.path, workspace_root / col2 / fed_workspace)

    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(exec, ['fx', 'aggregator', 'start'], workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(exec, ['fx', 'collaborator', 'start', '-n', col1], dir1)

        dir2 = workspace_root / col2 / fed_workspace
        executor.submit(exec, ['fx', 'collaborator', 'start', '-n', col2], dir2)
    shutil.rmtree(workspace_root)
