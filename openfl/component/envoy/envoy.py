# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Envoy module."""
import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tarfile import TarFile

import aiodocker
import yaml
from aiodocker import Docker
from aiodocker.containers import DockerContainer

from openfl.transport.grpc.director_client import ShardDirectorClient

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
                await self._run_collaborator(
                    experiment_name=experiment_name.lower(),
                    data_file_path=data_file_path,
                )

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
            cuda_devices_info = self._get_cuda_device_info()
            timeout = self.director_client.send_health_check(
                envoy_name=self.name,
                is_experiment_running=self.is_experiment_running,
                cuda_devices_info=cuda_devices_info,
            )
            time.sleep(timeout)

    def _get_cuda_device_info(self):
        cuda_devices_info = None
        try:
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
        except Exception as exc:
            logger.exception(f'Failed to get cuda device info: {exc}. '
                             f'Check your cuda device monitor plugin.')
        return cuda_devices_info

    async def _run_collaborator(self, experiment_name: str, data_file_path: Path):
        """Run the collaborator for the experiment running."""
        docker = aiodocker.Docker()
        docker_context_path = _create_docker_context(
            data_file_path=data_file_path,
            shard_descriptor_config=self.shard_descriptor_config,
        )
        await _build_docker_image(
            docker=docker,
            docker_context_path=docker_context_path,
            tag=experiment_name,
        )

        cuda_devices = ','.join(map(str, self.cuda_devices))

        cmd = (
            f'python run.py --name {self.name} '
            f'--plan_path plan/plan.yaml '
            f'--root_certificate {self.root_certificate} '
            f'--private_key {self.private_key} '
            f'--certificate {self.certificate} '
            f'--shard_config shard_descriptor_config.yaml '
            f'--cuda_devices {cuda_devices}'
        )

        container = await _create_docker_container(
            docker=docker,
            name=experiment_name,
            cmd=cmd,
        )

        await _start_and_monitor_docker_container(
            docker=docker,
            container=container
        )

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


def _create_docker_context(data_file_path: Path, shard_descriptor_config) -> Path:
    with TarFile(name=data_file_path, mode='a') as tar_file:
        envoy_module_path = Path(__file__)
        openfl_root_path = envoy_module_path.parent.parent.parent.parent.absolute()
        docker_file_path = openfl_root_path / 'openfl-docker' / 'Dockerfile.collaborator'
        run_collaborator_path = envoy_module_path.parent.absolute() / 'run_collaborator.py'

        with open('shard_descriptor_config.yaml', 'w') as f:
            yaml.dump(shard_descriptor_config, f)
        tar_file.add('shard_descriptor_config.yaml', 'shard_descriptor_config.yaml')
        os.remove('shard_descriptor_config.yaml')

        tar_file.add(docker_file_path, 'Dockerfile')
        tar_file.add(run_collaborator_path, 'run.py')

        template = shard_descriptor_config['template']
        module_path = template.split('.')[:-1]
        module_path[-1] = f'{module_path[-1]}.py'
        shar_descriptor_path = str(Path.joinpath(Path('.'), *module_path))
        tar_file.add(shar_descriptor_path, shar_descriptor_path)
    return data_file_path


async def _build_docker_image(docker: Docker, docker_context_path: Path, tag: str) -> None:
    with open(docker_context_path, 'rb') as f:
        fileobj = BytesIO(f.read())
        build_image_iter = docker.images.build(
            fileobj=fileobj,
            encoding='gzip',
            tag=tag,
            stream=True,
        )
    async for message in build_image_iter:
        if 'stream' not in message or len(message) > 1:
            print(message)
        logger.info(f'DOCKER BUILD {message.get("stream")}')


async def _create_docker_container(docker: Docker, name: str, cmd: str) -> DockerContainer:
    return await docker.containers.create_or_replace(
        config={
            'Cmd': ['/bin/bash', '-c', cmd],
            'Image': f'{name}:latest',
            'HostConfig': {
                'NetworkMode': 'host',
                'DeviceRequests': [{
                    'Driver': 'nvidia',
                    'Count': -1,
                    'Capabilities': [['gpu', 'compute', 'utility']],
                }],
                'ShmSize': 30 * 1024 * 1024 * 1024,
            },
        },
        name=name,
    )


async def _start_and_monitor_docker_container(docker: Docker, container: DockerContainer) -> None:
    subscriber = docker.events.subscribe()
    await container.start()
    logs_stream = container.log(stdout=True, stderr=True, follow=True)
    async for log in logs_stream:
        logger.info(f'CONTAINER {log}')
    while True:
        event = await subscriber.get()
        if event is None:
            break

        action = event.get('Action')

        if event['Actor']['ID'] == container._id:
            if action == 'stop':
                await container.delete(force=True)
                logger.info(f'=> deleted {container._id[:12]}')
            elif action == 'destroy':
                logger.info('=> done with this container!')
                break
            elif action == 'die':
                logger.info('=> container is died')
                break
