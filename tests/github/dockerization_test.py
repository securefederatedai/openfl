# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import socket
from subprocess import check_call
import shutil
import os
from pathlib import Path
import tarfile
from concurrent.futures import ProcessPoolExecutor


def create_signed_cert_for_collaborator(col, data_path):
    '''
    We do certs exchage for all participants in a single workspace to speed up this test run.
    Do not do this in real experiments in untrusted environments
    '''
    print(f'Certifying collaborator {col} with data path {data_path}...')
    # Create collaborator certificate request
    check_call([
        'fx', 'collaborator', 'generate-cert-request', '-d', data_path, '-n', col, '--silent'
    ])
    # Sign collaborator certificate
    check_call([
        'fx',
        'collaborator',
        'certify',
        '--request-pkg',
        f'col_{col}_to_agg_cert_request.zip',
        '--silent'
    ])

    # Pack the collaborators private key and the signed cert
    # as well as it's data.yaml to a tarball
    tarfiles = ['plan/data.yaml', f'agg_to_col_{col}_signed_cert.zip']
    with os.scandir('cert/client') as iterator:
        for entry in iterator:
            if entry.name.endswith('key'):
                tarfiles.append(entry.path)
    with tarfile.open(f'cert_col_{col}.tar', 'w') as t:
        for f in tarfiles:
            t.add(f)
    for f in tarfiles:
        os.remove(f)
    # Remove request archive
    os.remove(f'col_{col}_to_agg_cert_request.zip')


if __name__ == '__main__':
    # 1. Create the workspace
    parser = ArgumentParser()

    parser.add_argument('--template', default='keras_cnn_mnist')  # ['torch_cnn_mnist', 'keras_cnn_mnist']
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')  # This can be whatever unique directory name you want
    parser.add_argument('--col', default='one123dragons')
    parser.add_argument('--data_path', default='1')
    parser.add_argument('--base_image_tag', default='openfl')
    args = parser.parse_args()
    base_image_tag = args.base_image_tag
    fed_workspace = args.fed_workspace
    col = args.col

    # If an aggregator container will run on another machine
    # a relevant FQDN should be provided
    fqdn = socket.getfqdn()
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
    with ProcessPoolExecutor() as executor:
        executor.submit(lambda: check_call([
            'docker', 'run', '--rm',
            '--network', 'host',
            '-v', f'{Path.cwd().resolve()}/{aggregator_required_files}:/certs.tar',
            '-e', '\"CONTAINER_TYPE=aggregator\"',
            workspace_image_name,
            'bash /openfl/openfl-docker/start_actor_in_container.sh'])
        )
        executor.submit(lambda: check_call([
            'docker', 'run', '--rm',
            '--network', 'host',
            '-v', f'{Path.cwd()}/cert_col_{args.col}.tar:/certs.tar',
            '-e', '\"CONTAINER_TYPE=collaborator\"',
            '-e', f'\"COL={col}\"',
            workspace_image_name,
            'bash /openfl/openfl-docker/start_actor_in_container.sh'
        ]))

    # If containers are started but collaborator will fail to
    # conect the aggregator, the pipeline will go to the infinite loop