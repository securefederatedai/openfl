from importlib import import_module
from pathlib import Path
import logging

import yaml

from openfl.federated import Plan
from click import echo
import argparse

logger = logging.getLogger(__name__)

PLAN_PATH_DEFAULT = 'plan/plan.yaml'


def _parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', type=str, nargs=1)
    parser.add_argument('--plan_path', type=str, nargs='?', default=PLAN_PATH_DEFAULT)
    parser.add_argument('--root_certificate', type=str, nargs='?', default=None)
    parser.add_argument('--private_key', type=str, nargs='?', default=None)
    parser.add_argument('--certificate', type=str, nargs='?', default=None)
    parser.add_argument('--shard_config', type=str, nargs='?', default=None)
    parser.add_argument('--cuda_devices', type=str, nargs='?', default='cpu')
    return parser.parse_args()


def _run_collaborator(
        plan_path: str,
        name: str,
        root_certificate: str,
        private_key: str,
        certificate: str,
        shard_descriptor: str,
        cuda_devices: str,
) -> None:
    plan = Plan.parse(plan_config_path=Path(plan_path))
    echo(f'Data = {plan.cols_data_paths}')
    logger.info('🧿 Starting a Collaborator Service.')

    col = plan.get_collaborator(name, root_certificate, private_key,
                                certificate, shard_descriptor=shard_descriptor)
    col.set_available_devices(cuda=cuda_devices)
    col.run()


def _shard_descriptor_from_config(shard_config: dict):
    template = shard_config.get('template')
    if not template:
        raise Exception('You should define a shard '
                        'descriptor template in the envoy config')
    class_name = template.split('.')[-1]
    module_path = '.'.join(template.split('.')[:-1])
    params = shard_config.get('params', {})

    module = import_module(module_path)
    instance = getattr(module, class_name)(**params)

    return instance


if __name__ == '__main__':
    args = _parse_args()
    with open(args.shard_config) as f:
        shard_descriptor_config = yaml.safe_load(f)
    shard_descriptor = _shard_descriptor_from_config(shard_descriptor_config)
    _run_collaborator(
        plan_path=args.plan_path,
        name=args.name,
        root_certificate=args.root_certificate,
        private_key=args.private_key,
        certificate=args.certificate,
        shard_descriptor=shard_descriptor,
        cuda_devices=args.cuda_devices,
    )
