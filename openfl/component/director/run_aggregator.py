# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Script to run aggregator in docker."""

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np

from openfl.federated import Plan
from openfl.interface.cli import setup_logging

logger = logging.getLogger(__name__)

setup_logging()

PLAN_PATH_DEFAULT = 'plan/plan.yaml'


def _parse_args():
    parser = argparse.ArgumentParser(description='Run aggregator.')
    parser.add_argument('--plan_path', type=str, nargs='?', default=PLAN_PATH_DEFAULT)

    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--director_host', type=str)
    parser.add_argument('--director_port', type=int)

    parser.add_argument('--collaborators', type=str, nargs='+')
    parser.add_argument('--init_tensor_dict_path', type=str)
    parser.add_argument('--root_certificate', type=str, nargs='?', default=None)
    parser.add_argument('--private_key', type=str, nargs='?', default=None)
    parser.add_argument('--certificate', type=str, nargs='?', default=None)
    parser.add_argument('--tls', dest='tls', action='store_true')
    parser.add_argument('--no-tls', dest='tls', action='store_false')
    parser.set_defaults(tls=True)

    return parser.parse_args()


async def main(
        plan_path,
        experiment_name,
        director_host,
        director_port,
        collaborators,
        init_tensor_dict_path,
        root_certificate,
        certificate,
        private_key,
        tls,
):
    """Run main function."""
    plan = Plan.parse(plan_config_path=Path(plan_path))
    plan.authorized_cols = list(collaborators)

    logger.info('🧿 Starting the Aggregator Service.')
    init_tensor_dict = np.load(init_tensor_dict_path, allow_pickle=True)
    aggregator_grpc_server = plan.interactive_api_get_server(
        experiment_name=experiment_name,
        director_host=director_host,
        director_port=director_port,
        tensor_dict=init_tensor_dict,
        root_certificate=root_certificate,
        certificate=certificate,
        private_key=private_key,
        tls=tls,
    )

    logger.info('🧿 Starting the Aggregator Service.')
    grpc_server = aggregator_grpc_server.get_server()
    grpc_server.start()
    logger.info('Starting Aggregator gRPC Server')

    try:
        while not aggregator_grpc_server.aggregator.all_quit_jobs_sent():
            # Awaiting quit job sent to collaborators
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        grpc_server.stop(0)
        # Temporary solution to free RAM used by TensorDB
        aggregator_grpc_server.aggregator.tensor_db.clean_up(0)


if __name__ == '__main__':
    args = _parse_args()
    asyncio.run(main(
        plan_path=args.plan_path,
        experiment_name=args.experiment_name,
        director_host=args.director_host,
        director_port=args.director_port,
        collaborators=args.collaborators,
        init_tensor_dict_path=args.init_tensor_dict_path,
        root_certificate=args.root_certificate,
        certificate=args.certificate,
        private_key=args.private_key,
        tls=args.tls,
    ))
