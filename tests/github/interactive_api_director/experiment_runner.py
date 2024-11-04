import os
import logging
import shutil
import subprocess
import typing
from time import sleep
from dataclasses import dataclass
from rich.console import Console
from rich.logging import RichHandler


root = logging.getLogger()
root.setLevel(logging.INFO)
console = Console(width=160)
handler = RichHandler(console=console)
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)
handler.setFormatter(formatter)
root.addHandler(handler)

logger = logging.getLogger(__name__)


def prepare_collaborator_workspace(col_dir, arch_path):
    logger.info(f'Prepare collaborator directory: {col_dir}')
    if os.path.exists(col_dir):
        shutil.rmtree(col_dir)
    os.makedirs(col_dir)
    arch_col_path = shutil.copy(arch_path, col_dir)
    shutil.unpack_archive(arch_col_path, col_dir)
    logger.info('Collaborator directory prepared')


def run_aggregator(model_interface, fl_experiment):
    logger.info('Aggregator has been started.')
    fl_experiment.start_experiment(model_interface)
    logger.info('Aggregator has been stopped.')


def run_experiment(col_data_paths, model_interface, arch_path, fl_experiment):
    logger.info('Starting the experiment!')
    for col_dir in col_data_paths:
        prepare_collaborator_workspace(col_dir, arch_path)

    processes = []
    for col_name in col_data_paths:
        logger.info(f'Starting collaborator: {col_name}')
        p = subprocess.Popen(
            f"fx collaborator start -n {col_name} -p plan/plan.yaml -d data.yaml".split(' '),
            cwd=os.path.join(os.getcwd(), col_name)
        )
        processes.append(p)

    run_aggregator(model_interface, fl_experiment)
    for p in processes:
        p.terminate()

    logger.info('The experiment completed!')


def create_director(director_path, recreate, config):
    logger.info(f'Creating the director in {director_path}!')
    if os.path.exists(director_path):
        if not recreate:
            return
        shutil.rmtree(director_path)
    subprocess.Popen(
        f'fx director create-workspace -p {director_path}',
        shell=True
    ).wait()
    shutil.copy(config, director_path)


def create_envoy(col_path, recreate, envoy_config, shard_descriptor):
    logger.info(f'Creating the envoy in {col_path}!')
    if os.path.exists(col_path):
        if not recreate:
            return
        shutil.rmtree(col_path)
    subprocess.Popen(
        f'fx envoy create-workspace -p {col_path}',
        shell=True
    ).wait()
    shutil.copy(envoy_config, col_path)
    shutil.copy(shard_descriptor, col_path)


def create_federation(
    director_path: str,
    collaborator_paths: typing.Iterable[str],
    director_config,
    envoy_config,
    shard_descriptor,
    recreate=False
):
    logger.info('Creating the federation!')
    create_director(director_path, recreate, director_config)
    for col_path in collaborator_paths:
        create_envoy(col_path, recreate, envoy_config, shard_descriptor)
    # TODO: create mTLS
    logger.info('Federation was created')


@dataclass
class Shard:
    shard_name: str
    director_host: str
    director_port: int
    data_path: str


def run_federation(shards: typing.Dict[str, Shard], director_path: str):
    logger.info('Starting the experiment!')
    running_processes = []
    p = subprocess.Popen(
        "fx director start --disable-tls",
        shell=True,
        cwd=os.path.join(director_path)
    )
    sleep(2)
    running_processes.append(p)
    for collaborator_path, shard in shards.items():
        p = subprocess.Popen(
            f'fx envoy start '
            f'-n {shard.shard_name} -dh {shard.director_host} '
            f'-dp {shard.director_port} -p {shard.data_path}',
            shell=True,
            cwd=os.path.join(collaborator_path)
        )
        running_processes.append(p)
    logger.info('The federation started!')
    return running_processes


def stop_federation(running_processes):
    logger.info('Stopping the federation')
    for p in running_processes:
        p.terminate()
    logger.info('Federation was stopped')
