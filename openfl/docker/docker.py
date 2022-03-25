# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Docker module."""

from dataclasses import dataclass
import logging
import os
from io import BytesIO
from pathlib import Path
from tarfile import TarFile
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import field

import aiodocker
import yaml
from aiodocker.containers import DockerContainer

logger = logging.getLogger(__name__)

OPENFL_ROOT_PATH = Path(__file__).parent.parent.parent.absolute()


@dataclass
class DockerConfig:
    use_docker: bool = True
    env: Dict[str, str] = field(default_factory=dict)
    buildargs: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)

    @property
    def binds(self) -> List[str]:
        """Convert docker volumes to binds."""
        binds = []
        for volume in map(lambda x: x.split(':'), self.volumes):
            target = volume[0]
            bind = volume[0] if len(volume) == 1 else volume[1]

            if bind.startswith('./'):
                bind = bind.replace('.', '/code', 1)
            elif bind.startswith('~'):
                bind = bind.replace('~', '/root', 1)

            if target.startswith('~'):
                target = str(Path(target).expanduser().resolve())
            if target.startswith('./'):
                target = str(Path(target).expanduser().resolve())
            binds.append(f'{target}:{bind}')

        return binds

    @property
    def env_list(self):
        env_lst = []
        for k, v in self.env.items():
            if v is None:
                continue
            env_lst.append(f'{k}={v}')
        return env_lst


class Docker:
    """Docker class."""

    def __init__(
            self, *,
            config: Optional[DockerConfig] = None,
    ):
        """Initialize an docker object."""
        self.docker = aiodocker.Docker()
        if config is None:
            config = DockerConfig()
        self.config = config

    async def build_image(
            self, *,
            context_path: Path,
            tag: str,
    ) -> str:
        """Build docker image."""
        with open(context_path, 'rb') as f:
            fileobj = BytesIO(f.read())
            build_image_iter = self.docker.images.build(
                fileobj=fileobj,
                encoding='gzip',
                tag=tag,
                stream=True,
                buildargs=self.config.buildargs,
            )
        async for message in build_image_iter:
            if 'stream' not in message or len(message) > 1:
                print(message)
            logger.info(f'DOCKER BUILD {message.get("stream")}')
        return f'{tag}:latest'

    async def create_container(
            self, *,
            name: str,
            image_tag: str,
            cmd: str,
            gpu_allowed: bool = False,
    ) -> DockerContainer:
        """Create docker container."""
        config = {
            'Cmd': ['/bin/bash', '-c', cmd],
            'Env': self.config.env_list,
            'Image': image_tag,
            'HostConfig': {
                'NetworkMode': 'host',
                'Binds': self.config.binds,
                'ShmSize': 30 * 1024 * 1024 * 1024,
            },
        }
        if gpu_allowed:
            config['HostConfig'].update(**{
                'DeviceRequests': [{
                    'Driver': 'nvidia',
                    'Count': -1,
                    'Capabilities': [['gpu', 'compute', 'utility']],
                }],
            })

        logger.info(f'{config=}')

        return await self.docker.containers.create_or_replace(
            config=config,
            name=name,
        )

    async def start_and_monitor_container(
            self, *,
            container: DockerContainer,
    ) -> None:
        """Start and monitor docker container."""
        subscriber = self.docker.events.subscribe()
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
                    logger.info(f'=> deleted {container._id[:12]}')
                    break
                elif action == 'destroy':
                    logger.info('=> done with this container!')
                    break
                elif action == 'die':
                    logger.info('=> container is died')
                    break


def create_aggregator_context(data_file_path: Path, init_tensor_dict_path: Path):
    """Create context for aggregator service."""
    with TarFile(name=data_file_path, mode='a') as tar_file:
        docker_file_path = OPENFL_ROOT_PATH / 'openfl-docker' / 'Dockerfile.aggregator'
        run_aggregator_path = (OPENFL_ROOT_PATH / 'openfl' / 'component' / 'director'
                               / 'run_aggregator.py')
        tar_file.add(docker_file_path, 'Dockerfile')
        tar_file.add(run_aggregator_path, 'run.py')
        tar_file.add(init_tensor_dict_path, 'init_tensor_dict.pickle')
    return data_file_path


def create_collaborator_context(
        data_file_path: Path,
        shard_descriptor_config
) -> Path:
    """Create context for collaborator service."""
    with TarFile(name=data_file_path, mode='a') as tar_file:
        docker_file_path = OPENFL_ROOT_PATH / 'openfl-docker' / 'Dockerfile.collaborator'
        run_collaborator_path = (OPENFL_ROOT_PATH / 'openfl' / 'component' / 'envoy'
                                 / 'run_collaborator.py')

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
