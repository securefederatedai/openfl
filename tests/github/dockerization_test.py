# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import socket
from subprocess import check_call
import shutil
import os
from pathlib import Path
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

from tests.github.utils import start_aggregator_container
from tests.github.utils import start_collaborator_container
from tests.github.utils import create_signed_cert_for_collaborator
from openfl.utilities.utils import getfqdn_env

if __name__ == '__main__':
    # 1. Create the workspace
    parser = ArgumentParser()
    workspace_choice = []
    with os.scandir('openfl-workspace') as iterator:
        for entry in iterator:
            if entry.name not in ['__init__.py', 'workspace', 'default']:
                workspace_choice.append(entry.name)
    parser.add_argument('--template', default='keras_cnn_mnist', choices=workspace_choice)
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')
    parser.add_argument('--col', default='one123dragons')
    parser.add_argument('--data_path', default='1')
    parser.add_argument('--base_image_tag', default='openfl')
    args = parser.parse_args()
    base_image_tag = args.base_image_tag
    fed_workspace = args.fed_workspace
    col = args.col

    # If an aggregator container will run on another machine
    # a relevant FQDN should be provided
    fqdn = getfqdn_env()
    # Build base image
    check_call([
        'docker', 'build', '-t', base_image_tag, '-f', 'openfl-docker/Dockerfile.base', '.'
    ])

    # Create FL workspace
    shutil.rmtree(fed_workspace, ignore_errors=True)
    check_call([
        'fx', 'workspace', 'create', '--prefix', fed_workspace, '--template', args.template
    ])
    os.chdir(fed_workspace)
    fed_directory = Path().resolve()  # Get the absolute directory path for the workspace

    # Initialize FL plan
    check_call(['fx', 'plan', 'initialize', '-a', fqdn])

    # 2. Build the workspace image and save it to a tarball

    # This commant builds an image tagged $FED_WORKSPACE
    # Then it saves it to a ${FED_WORKSPACE}_image.tar

    check_call(['fx', 'workspace', 'dockerize', '--base_image', base_image_tag])

    # We remove the base OpenFL image as well
    # as built workspace image to simulate starting
    # on another machine
    workspace_image_name = fed_workspace
    check_call(['docker', 'image', 'rm', '-f', base_image_tag, workspace_image_name])

    # 3. Generate certificates for the aggregator and the collaborator

    # Create certificate authority for the workspace
    check_call(['fx', 'workspace', 'certify'])

    # Prepare a tarball with the collab's private key, the singed cert,
    # and data.yaml for collaborator container
    # This step can be repeated for each collaborator
    create_signed_cert_for_collaborator(args.col, args.data_path)

    # Also perform certificate generation for the aggregator.
    # Create aggregator certificate
    check_call(['fx', 'aggregator', 'generate-cert-request', '--fqdn', fqdn])
    # Sign aggregator certificate
    # Remove '--silent' if you run this manually
    check_call(['fx', 'aggregator', 'certify', '--fqdn', fqdn, '--silent'])

    # Pack all files that aggregator need to start training
    aggregator_required_files = 'cert_agg.tar'
    with tarfile.open(aggregator_required_files, 'w') as f:
        for d in ['plan', 'cert', 'save']:
            f.add(d)
            shutil.rmtree(d)

    # 4. Load the image
    image_tar = f'{fed_workspace}_image.tar'
    check_call(['docker', 'load', '--input', image_tar])
    time.sleep(5)
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.submit(start_aggregator_container, args=(
            workspace_image_name,
            aggregator_required_files
        ))
        time.sleep(5)
        executor.submit(start_collaborator_container, args=(
            workspace_image_name,
            col
        ))
    # If containers are started but collaborator will fail to
    # conect the aggregator, the pipeline will go to the infinite loop
