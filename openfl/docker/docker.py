import logging
import os
from io import BytesIO
from pathlib import Path
from tarfile import TarFile
from typing import List
from typing import Optional

import aiodocker
import yaml
from aiodocker.containers import DockerContainer

logger = logging.getLogger(__name__)


class Docker:

    def __init__(self):
        self.docker = aiodocker.Docker()

    async def build_image(
            self, *,
            context_path: Path,
            tag: str,
    ) -> str:
        with open(context_path, 'rb') as f:
            fileobj = BytesIO(f.read())
            build_image_iter = self.docker.images.build(
                fileobj=fileobj,
                encoding='gzip',
                tag=tag,
                stream=True,
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
            volumes: Optional[List[str]],
            gpu_allowed: bool = False,
    ) -> DockerContainer:
        if volumes is None:
            volumes = []

        binds = volumes_to_binds(volumes)

        # volumes = [
        #     f'{target}:{bind}'
        #     for target, bind in map(lambda x: x.split(':'), volumes)
        # ]
        config = {
            'Cmd': ['/bin/bash', '-c', cmd],
            'Image': image_tag,
            # 'Volumes': {
            #     target: {
            #         'bind': bind,
            #         'mode': 'r',
            #     } for target, bind in map(lambda x: x.split(':'), volumes)
            # },
            'HostConfig': {
                'NetworkMode': 'host',
                'Binds': binds,
                # 'Mounts': [{
                #     "Target": "path/in/the/container",
                #     "Source": "volumeName",
                #     "Type": "volume",
                #     "ReadOnly": False,
                # }]
            },
        }
        if gpu_allowed:
            config['HostConfig'].update(**{
                'DeviceRequests': [{
                    'Driver': 'nvidia',
                    'Count': -1,
                    'Capabilities': [['gpu', 'compute', 'utility']],
                }],
                'ShmSize': 30 * 1024 * 1024 * 1024,
            })

        return await self.docker.containers.create_or_replace(
            config=config,
            name=name,
        )

    async def start_and_monitor_container(
            self, *,
            container: DockerContainer,
    ) -> None:
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
                    await container.delete(force=True)
                    logger.info(f'=> deleted {container._id[:12]}')
                elif action == 'destroy':
                    logger.info('=> done with this container!')
                    break
                elif action == 'die':
                    logger.info('=> container is died')
                    break


def create_collaborator_context(
        data_file_path: Path,
        shard_descriptor_config
) -> Path:
    with TarFile(name=data_file_path, mode='a') as tar_file:
        docker_module_path = Path(__file__)
        openfl_root_path = docker_module_path.parent.parent.parent.absolute()
        docker_file_path = openfl_root_path / 'openfl-docker' / 'Dockerfile.collaborator'
        run_collaborator_path = openfl_root_path / 'openfl' / 'component' / 'envoy' / 'run_collaborator.py'

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


def volumes_to_binds(volumes: List[str]) -> List[str]:
    binds = []
    for volume in map(lambda x: x.split(':'), volumes):
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
