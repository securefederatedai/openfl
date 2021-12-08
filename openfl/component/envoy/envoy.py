# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Envoy module."""
import asyncio
import logging
import shutil
from io import BytesIO

import aiodocker
from aiodocker import utils
import time
import uuid
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from click import echo

from openfl.federated import Plan
from openfl.transport.grpc.director_client import ShardDirectorClient
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)

DEFAULT_RETRY_TIMEOUT_IN_SECONDS = 5


class Envoy:
    """Envoy class."""

    def __init__(self, *, shard_name, director_host, director_port, shard_descriptor,
                 shard_descriptor_config,
                 root_certificate: str = None, private_key: str = None, certificate: str = None,
                 tls: bool = True, cuda_devices=(), cuda_device_monitor=None) -> None:
        """Initialize a envoy object."""
        self.name = shard_name
        self.root_certificate = Path(
            root_certificate).absolute() if root_certificate is not None else None
        self.private_key = Path(private_key).absolute() if root_certificate is not None else None
        self.certificate = Path(certificate).absolute() if root_certificate is not None else None
        self.director_client = ShardDirectorClient(
            director_host=director_host,
            director_port=director_port,
            shard_name=shard_name,
            tls=tls,
            root_certificate=root_certificate,
            private_key=private_key,
            certificate=certificate
        )

        self.shard_descriptor = shard_descriptor
        self.shard_descriptor_config = shard_descriptor_config
        self.cuda_devices = tuple(cuda_devices)

        # Optional plugins
        self.cuda_device_monitor = cuda_device_monitor

        self.executor = ThreadPoolExecutor()
        self.running_experiments = {}
        self.is_experiment_running = False
        self._health_check_future = None

    async def run(self):
        """Run of the envoy working cycle."""
        while True:
            try:
                # Workspace import should not be done by gRPC client!
                experiment_name = self.director_client.wait_experiment()
                data_stream = self.director_client.get_experiment_data(experiment_name)
            except Exception as exc:
                logger.exception(f'Failed to get experiment: {exc}')
                time.sleep(DEFAULT_RETRY_TIMEOUT_IN_SECONDS)
                continue
            data_file_path = self._save_data_stream_to_file(data_stream)
            self.is_experiment_running = True
            try:
                from tarfile import TarFile, TarInfo
                import os
                docker = aiodocker.Docker()
                with TarFile(name=data_file_path, mode='a') as tar_file:
                    docker_file_path = Path(
                        __file__).parent.parent.parent.parent.absolute() / 'openfl-docker' / 'Dockerfile.collaborator'
                    run_collaborator_path = Path(
                        __file__).parent.absolute() / 'run_collaborator.py'
                    with open('shard_descriptor_config.yaml', 'w') as f:
                        yaml.dump(self.shard_descriptor_config, f)
                    tar_file.add(docker_file_path, 'Dockerfile')
                    tar_file.add(run_collaborator_path, 'run.py')
                    tar_file.add('shard_descriptor_config.yaml', 'shard_descriptor_config.yaml')
                    # os.remove('shard_descriptor_config.yaml')

                with open(data_file_path, 'rb') as f:
                    fileobj = BytesIO(f.read())
                    build_image_iter = docker.images.build(
                        fileobj=fileobj,
                        encoding='gzip',
                        tag=experiment_name,
                        stream=True,
                    )
                async for l in build_image_iter:
                    print(l)
                cmd = (
                    f'python run.py --name {self.name} '
                    f'--plan_path plan/plan.yaml '
                    f'--root_certificate {self.root_certificate} '
                    f'--private_key {self.private_key} '
                    f'--certificate {self.certificate} '
                    f'--shard_config shard_descriptor_config.yaml '
                    f'--cuda_devices cpu'
                )
                container = await docker.containers.create_or_replace(
                    config={
                        'Cmd': ['/bin/bash', '-c', cmd],
                        'Image': f'{experiment_name}:latest',
                    },
                    name=experiment_name,
                )
                subscriber = docker.events.subscribe()
                await container.start()
                while True:
                    event = await subscriber.get()
                    if event is None:
                        break

                    for key, value in event.items():
                        print(key, ":", value)

                    if event["Actor"]["ID"] == container._id:
                        if event["Action"] == "stop":
                            await container.delete(force=True)
                            print(f"=> deleted {container._id[:12]}")
                        elif event["Action"] == "destroy":
                            print("=> done with this container!")
                            break

            except Exception as exc:
                logger.exception(f'Collaborator failed with error: {exc}:')
            finally:
                self.is_experiment_running = False

    @staticmethod
    def _save_data_stream_to_file(data_stream):
        data_file_path = Path(str(uuid.uuid4())).absolute()
        with open(data_file_path, 'wb') as data_file:
            for response in data_stream:
                if response.size == len(response.npbytes):
                    data_file.write(response.npbytes)
                else:
                    raise Exception('Broken archive')
        return data_file_path

    def send_health_check(self):
        """Send health check to the director."""
        logger.info('The health check sender is started.')
        while True:
            # Need a separate method 'Get self state' or smth
            cuda_devices_info = None
            if self.cuda_device_monitor is not None:
                cuda_devices_info = []
                cuda_driver_version = self.cuda_device_monitor.get_driver_version()
                cuda_version = self.cuda_device_monitor.get_cuda_version()
                for device_id in self.cuda_devices:
                    memory_total = self.cuda_device_monitor.get_device_memory_total(device_id)
                    memory_utilized = self.cuda_device_monitor.get_device_memory_utilized(
                        device_id
                    )
                    device_utilization = self.cuda_device_monitor.get_device_utilization(device_id)
                    device_name = self.cuda_device_monitor.get_device_name(device_id)
                    cuda_devices_info.append({
                        'index': device_id,
                        'memory_total': memory_total,
                        'memory_utilized': memory_utilized,
                        'device_utilization': device_utilization,
                        'cuda_driver_version': cuda_driver_version,
                        'cuda_version': cuda_version,
                        'name': device_name,
                    })

            timeout = self.director_client.send_health_check(
                envoy_name=self.name,
                is_experiment_running=self.is_experiment_running,
                cuda_devices_info=cuda_devices_info,
            )
            time.sleep(timeout)

    def _run_collaborator(self, plan='plan/plan.yaml'):
        """Run the collaborator for the experiment running."""
        plan = Plan.parse(plan_config_path=Path(plan))

        # TODO: Need to restructure data loader config file loader
        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(self.name, self.root_certificate, self.private_key,
                                    self.certificate, shard_descriptor=self.shard_descriptor)
        col.set_available_devices(cuda=self.cuda_devices)
        col.run()

    def start(self):
        """Start the envoy."""
        try:
            is_accepted = self.director_client.report_shard_info(
                shard_descriptor=self.shard_descriptor,
                cuda_devices=self.cuda_devices)
        except Exception as exc:
            logger.exception(f'Failed to report shard info: {exc}')
        else:
            if is_accepted:
                # Shard accepted for participation in the federation
                logger.info('Shard accepted')
                self._health_check_future = self.executor.submit(self.send_health_check)
                asyncio.run(self.run())
            else:
                # Shut down
                logger.error('Report shard info was not accepted')
