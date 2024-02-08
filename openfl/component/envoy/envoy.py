# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Envoy module."""

import logging
import time
import traceback
import uuid
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union

from openfl.federated import Plan
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.plugins.processing_units_monitor.cuda_device_monitor import CUDADeviceMonitor
from openfl.transport.grpc.exceptions import ShardNotFoundError
from openfl.transport.grpc.director_client import ShardDirectorClient
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)

DEFAULT_RETRY_TIMEOUT_IN_SECONDS = 5


class Envoy:
    """Envoy class."""

    def __init__(
            self, *,
            shard_name: str,
            director_host: str,
            director_port: int,
            shard_descriptor: Type[ShardDescriptor],
            root_certificate: Optional[Union[Path, str]] = None,
            private_key: Optional[Union[Path, str]] = None,
            certificate: Optional[Union[Path, str]] = None,
            tls: bool = True,
            install_requirements: bool = True,
            cuda_devices: Union[tuple, list] = (),
            cuda_device_monitor: Optional[Type[CUDADeviceMonitor]] = None,
            review_plan_callback: Union[None, Callable] = None,
    ) -> None:
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
        self.cuda_devices = tuple(cuda_devices)
        self.install_requirements = install_requirements

        self.review_plan_callback = review_plan_callback

        # Optional plugins
        self.cuda_device_monitor = cuda_device_monitor

        self.executor = ThreadPoolExecutor()
        self.running_experiments = {}
        self.is_experiment_running = False
        self._health_check_future = None

    def run(self):
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

            try:
                with ExperimentWorkspace(
                        experiment_name=f'{self.name}_{experiment_name}',
                        data_file_path=data_file_path,
                        install_requirements=self.install_requirements
                ):
                    # If the callback is passed
                    if self.review_plan_callback:
                        # envoy to review the experiment before starting
                        if not self.review_plan_callback('plan', 'plan/plan.yaml'):
                            self.director_client.set_experiment_failed(
                                experiment_name,
                                error_description='Experiment is rejected'
                                f' by Envoy "{self.name}" manager.'
                            )
                            continue
                        logger.debug(
                            f'Experiment "{experiment_name}" was accepted by Envoy manager'
                        )
                    self.is_experiment_running = True
                    self._run_collaborator()
            except Exception as exc:
                logger.exception(f'Collaborator failed with error: {exc}:')
                self.director_client.set_experiment_failed(
                    experiment_name,
                    error_code=1,
                    error_description=traceback.format_exc()
                )
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
        logger.info('Sending health check to director.')
        timeout = DEFAULT_RETRY_TIMEOUT_IN_SECONDS
        while True:
            cuda_devices_info = self._get_cuda_device_info()
            try:
                timeout = self.director_client.send_health_check(
                    envoy_name=self.name,
                    is_experiment_running=self.is_experiment_running,
                    cuda_devices_info=cuda_devices_info,
                )
            except ShardNotFoundError:
                logger.info('The director has lost information about current shard. Resending...')
                self.director_client.report_shard_info(
                    shard_descriptor=self.shard_descriptor,
                    cuda_devices=self.cuda_devices
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

    def _run_collaborator(self, plan='plan/plan.yaml'):
        """Run the collaborator for the experiment running."""
        plan = Plan.parse(plan_config_path=Path(plan))

        # TODO: Need to restructure data loader config file loader
        logger.info(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting  the Collaborator Service.')

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
            sys.exit(1)
        else:
            if is_accepted:
                # Shard accepted for participation in the federation
                logger.info('Shard accepted')
                self._health_check_future = self.executor.submit(self.send_health_check)
                self.run()
            else:
                # Shut down
                logger.error('Report shard info was not accepted')
                sys.exit(1)
