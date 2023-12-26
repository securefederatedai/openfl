# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import horovod.torch as hvd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.ptglue_inmemory import InHorovodGlueMrpcFederatedDataLoader
from src.InHorovodLLMTaskRunner import InHorovodLLMTaskRunner
import json
from logging import getLogger
import traceback
import openfl.native as fx

def get_args():
    """
    Get command-line arguments for a script.

    Parameters:
    - data_path (str): Path to the data.
    - model_path (str): Path to the model.

    Returns:
    - args (Namespace): A namespace containing the parsed arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument(
        "--data_path", type=str, help="Path to the data.", required=True
    )
    parser.add_argument("--out_path", type=str, help="Path to the data.", required=True)
    parser.add_argument(
        "--state_path", type=str, help="Path to the model.", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, help="Path to the model.", required=True
    )
    parser.add_argument("--kwargs", type=str, help="Path to the model.", required=True)
    parser.add_argument("--func", type=str, help="Path to the model.", required=True)

    args = parser.parse_args()
    return args


def main():
    logger = getLogger(__name__)
    fx.setup_logging(level='INFO', log_file=None)
    try:
        logger.info(f"starting horovod")
        hvd.init()
        logger.info(f"started global node:local node, {hvd.rank()}, {hvd.local_rank()}")
        logger.info('getting arguments')
        args = get_args()
        logger.info('loading data')
        data_loader = InHorovodGlueMrpcFederatedDataLoader(
            data_path=args.data_path, batch_size=args.batch_size
        )
        logger.info('get taskrunner')
        taskrunner = InHorovodLLMTaskRunner(data_loader)
        func = getattr(taskrunner, args.func)
        kwargs = json.loads(args.kwargs)
        kwargs.update(
            {
                "data_path": args.data_path,
                "state_path": args.state_path,
                "out_path": args.out_path,
            }
        )
        logger.info(f"running funtion {args.func}")
        p = func(**kwargs)
        return p
    except:
        logger.error(traceback.print_exc())


if __name__ == "__main__":
    main()
