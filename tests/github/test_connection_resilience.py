# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from subprocess import check_call
import asyncio
import shutil
import os
from pathlib import Path
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

from tests.github.utils import create_signed_cert_for_collaborator
from tests.github.utils import edit_plan_yaml
from tests.github.utils import prepare_tc_flags

ROUNDS_TO_TRAIN = 2
DATA_REDUCTION_FACTOR = 10

if __name__ == '__main__':
    # 1. Create the workspace
    parser = ArgumentParser()
    workspace_choice = []
    with os.scandir('openfl-workspace') as iterator:
        for entry in iterator:
            if entry.name not in ['__init__.py', 'workspace', 'default']:
                workspace_choice.append(entry.name)
    docker_tc_help_string = (
        'options: "com.docker-tc.limit=1mbps" "com.docker-tc.delay=100ms" "com.docker-tc.loss=50%" '
        '"com.docker-tc.duplicate=50%" "com.docker-tc.corrupt=10%" '
        'read more: https://github.com/lukaszlach/docker-tc#usage'
    )
    parser.add_argument('--template', default='torch_unet_kvasir_gramine_ready',
                        choices=workspace_choice)
    parser.add_argument('--fed_workspace', default='test_federation')
    parser.add_argument('--reconnection_timeout', default='10', help='in seconds')
    parser.add_argument('--number_of_collaborators', type=int, default='1')
    parser.add_argument('--start_stage', type=int, default=1,
                        help='1 - start from building a base OpenFL image \n' +
                        '2 - start from creating a workspace \n' +
                        '3 - start from building a dockerized workspace image \n' +
                        '4 - run Federation')
    parser.add_argument(
        '-atc',
        '--aggregator_traffic_control_flag', action='append', default=[],
        help=docker_tc_help_string
        )
    parser.add_argument(
        '-ctc',
        '--collaborator_traffic_control_flag', action='append', default=[],
        help=docker_tc_help_string
        )
    args = parser.parse_args()
    fed_workspace = args.fed_workspace
    workspace_image_name = fed_workspace
    reconnection_timeout = args.reconnection_timeout
    n_cols = args.number_of_collaborators
    stage = args.start_stage
    col_names = [f'col_{i+1}' for i in range(n_cols)]
    col_data_paths = [str(i) for i in range(1, n_cols + 1)]
    base_image_tag = 'openfl'
    aggregator_container_name = 'aggregatorContainer'
    # collaborator_container_base_name = f'{fed_workspace}_collaborator_container'

    # traffic control flags for containers
    agg_tc_flags = prepare_tc_flags(args.aggregator_traffic_control_flag)
    col_tc_flags = prepare_tc_flags(args.collaborator_traffic_control_flag)

    # FQDN here is the name of Aggregator container
    fqdn = aggregator_container_name

    # Get the absolute directory path for the workspace
    fed_directory = Path().resolve() / fed_workspace

    if stage == 1:
        # Build base image
        check_call([
            'docker', 'build', '-t', base_image_tag, '-f', 'openfl-docker/Dockerfile.base', '.'
        ])

    if stage <= 2:
        # Create FL workspace
        shutil.rmtree(fed_directory, ignore_errors=True)
        check_call([
            'fx', 'workspace', 'create', '--prefix', fed_workspace, '--template', args.template
        ])
        os.chdir(fed_directory)

        # Chenging experiment settings
        plan_path = fed_directory / 'plan' / 'plan.yaml'
        settings_dict = {
            'aggregator.settings.rounds_to_train': ROUNDS_TO_TRAIN,
            'data_loader.settings.collaborator_count': n_cols * DATA_REDUCTION_FACTOR,
        }
        edit_plan_yaml(plan_path, settings_dict)

        # Initialize FL plan
        check_call(['fx', 'plan', 'initialize', '-a', fqdn])

    if stage <= 3:
        os.chdir(fed_directory)
        # 2. Build the workspace image
        # This commant builds an image tagged $fed_workspace
        check_call(['fx', 'workspace', 'dockerize', '--base_image', base_image_tag, '--no-save'])

    if stage <= 4:
        # 3. Generate certificates for the aggregator and the collaborator

        os.chdir(fed_directory)
        # Create certificate authority for the workspace
        check_call(['fx', 'workspace', 'certify'])

        # Then perform certificate generation for the aggregator.
        # Create aggregator certificate
        check_call(['fx', 'aggregator', 'generate-cert-request', '--fqdn', fqdn])
        # Sign aggregator certificate
        # Remove '--silent' if you run this manually
        check_call(['fx', 'aggregator', 'certify', '--fqdn', fqdn, '--silent'])
        # Pack all files that aggregator need to start training

        collaborator_required_files = []
        for i in range(n_cols):
            # Prepare a tarball with the collab's private key, the singed cert,
            # and data.yaml for collaborator container
            # This step can be repeated for each collaborator
            path_ = create_signed_cert_for_collaborator(col_names[i], col_data_paths[i])
            collaborator_required_files.append(path_)

        aggregator_required_files = fed_directory / 'cert_agg.tar'
        with tarfile.open(aggregator_required_files, 'w') as f:
            for d in ['plan', 'cert', 'save']:
                f.add(d)
                # shutil.rmtree(d)

        # 4. Run the Federation
        def get_start_aggregator_command():
            return ('docker run --rm '
                    f'--network {docker_network_name} '
                    f'-v {aggregator_required_files}:/certs.tar '
                    '-e \"CONTAINER_TYPE=aggregator\" '
                    f'{agg_tc_flags}'
                    f'--name {aggregator_container_name} '
                    f'{workspace_image_name} '
                    'bash /openfl/openfl-docker/start_actor_in_container.sh')

        def get_start_collaborator_command(index):
            return ('docker run --rm '
                    f'--network {docker_network_name} '
                    f'-v {collaborator_required_files[index]}:/certs.tar '
                    '-e \"CONTAINER_TYPE=collaborator\" '
                    f'-e \"no_proxy={aggregator_container_name}\" '
                    f'-e \"COL={col_names[index]}\" '
                    f'{col_tc_flags}'
                    f'--name {col_names[index]} '
                    f'{workspace_image_name} '
                    'bash /openfl/openfl-docker/start_actor_in_container.sh')

        async def run_federation_async():
            agg_proc = await asyncio.create_subprocess_shell(
                get_start_aggregator_command(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            asyncio.sleep(1)
            col_processes = []
            for i in range(n_cols):
                col_proc = await asyncio.create_subprocess_shell(
                    get_start_collaborator_command(i),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
                col_processes.append(col_proc)


        def run_federation_subprocess():
            with ProcessPoolExecutor(max_workers=n_cols + 1) as executor:
                executor.submit(
                    check_call, get_start_aggregator_command(), shell=True
                )
                time.sleep(3)
                for i in range(n_cols):
                    executor.submit(
                        check_call, get_start_collaborator_command(i), shell=True
                    )

        # Create a Docker Network
        docker_network_name = f'{fed_workspace}_network'
        check_call(f'docker network create {docker_network_name} || true', shell=True)

        run_federation_subprocess()
        try:
            pass
        except Exception as e:
            print(e)
        finally:
            check_call(
                f'docker stop {aggregator_container_name} {" ".join(col_names)} || true',
                shell=True)
            check_call(f'docker network rm {docker_network_name}', shell=True)
        # If containers are started but collaborator will fail to
        # conect the aggregator, the pipeline will go to the infinite loop
