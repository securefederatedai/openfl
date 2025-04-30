# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DirectorClient module."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union  # type: ignore

import grpc
from grpc._channel import _MultiThreadedRendezvous as DataStream

from openfl.experimental.workflow.protocols import director_pb2, director_pb2_grpc
from openfl.experimental.workflow.transport.grpc.exceptions import EnvoyNotFoundError
from openfl.protocols.utils import datastream_to_proto

from .grpc_channel_options import channel_options

logger = logging.getLogger(__name__)


class EnvoyDirectorClient:
    """Envoy director client class for envoys.

    This class communicates with the director to manage the envoys
    participation in the federation.

    Attributes:
        director_addr (host:port): Director Address
        envoy_name (Optional[str]): The name of the envoy.
        stub (director_pb2_grpc.DirectorStub): The gRPC stub for communication
            with the director.
    """

    def __init__(
        self,
        *,
        director_host: str,
        director_port: int,
        envoy_name: Optional[str] = None,
        tls: bool = False,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
    ) -> None:
        """
        Initialize director client object.

        Args:
            director_host (str): The host name for Director server.
            director_port (int): The port number for Director server.
            envoy_name (Optional[str]): The name of the envoy.
            tls (bool): Whether to use TLS for the connection.
            root_certificate (Optional[Union[Path, str]]): The path to the root certificate for the
                TLS connection.
            private_key (Optional[Union[Path, str]]): The path to the private key for the TLS
                connection.
            certificate (Optional[Union[Path, str]]): The path to the certificate for the TLS
                connection.
        """
        director_addr = f"{director_host}:{director_port}"
        self.envoy_name = envoy_name
        logger.info("Director address: %s", director_addr)
        if not tls:
            channel = grpc.insecure_channel(director_addr, options=channel_options)
        else:
            if not (root_certificate and private_key and certificate):
                raise Exception("No certificates provided for TLS connection")
            try:
                with open(root_certificate, "rb") as f:
                    root_certificate_b = f.read()
                with open(private_key, "rb") as f:
                    private_key_b = f.read()
                with open(certificate, "rb") as f:
                    certificate_b = f.read()
            except FileNotFoundError as exc:
                raise Exception(f"Provided certificate file is not exist: {exc.filename}")

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificate_b,
                private_key=private_key_b,
                certificate_chain=certificate_b,
            )
            channel = grpc.secure_channel(director_addr, credentials, options=channel_options)
        self.stub = director_pb2_grpc.DirectorStub(channel)

    def connect_envoy(self, envoy_name: str) -> bool:
        """Attempt to establish a connection with the director.
        Args:
            envoy_name (str): Name of the envoy

        Returns:
            response.accepted (bool): Whether Envoy connection is accepted or not
        """
        logger.info(f"Sending {envoy_name} connection request to director")

        request = director_pb2.SendConnectionRequest(envoy_name=envoy_name)
        response = self.stub.EnvoyConnectionRequest(request)

        return response.accepted

    def wait_experiment(self) -> str:
        """
        Waits for experiment data from the director.

        Returns:
            experiment_name (str): The name of the experiment.
        """
        logger.info("Waiting for an experiment to run...")
        response = self.stub.WaitExperiment(self._get_experiment_data())
        logger.info("New experiment received: %s", response)
        if not response.experiment_name:
            raise ValueError("No experiment name received")
        return response.experiment_name

    def get_experiment_data(self, experiment_name) -> DataStream:
        """
        Get an experiment data from the director.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            data_stream (grpc._channel._MultiThreadedRendezvous): The data
                stream of the experiment data.
        """
        logger.info("Getting experiment data for %s...", experiment_name)
        request = director_pb2.GetExperimentDataRequest(
            experiment_name=experiment_name, collaborator_name=self.envoy_name
        )
        data_stream = self.stub.GetExperimentData(request)

        return data_stream

    def _get_experiment_data(self) -> director_pb2.WaitExperimentRequest:
        """Generate the experiment data request.

        Returns:
            director_pb2.WaitExperimentRequest: The request for experiment
                data.
        """
        return director_pb2.WaitExperimentRequest(collaborator_name=self.envoy_name)

    def send_health_check(
        self,
        *,
        envoy_name: str,
        is_experiment_running: bool,
    ) -> int:
        """Send envoy health check.

        Args:
            envoy_name (str): The name of the envoy.
            is_experiment_running (bool): Whether an experiment is currently
                running.

        Returns:
            health_check_period (int): The period for health checks.
        """
        status = director_pb2.UpdateEnvoyStatusRequest(
            name=envoy_name,
            is_experiment_running=is_experiment_running,
        )

        logger.debug("Sending health check status: %s", status)
        try:
            response = self.stub.UpdateEnvoyStatus(status)
        except grpc.RpcError as rpc_error:
            logger.error(rpc_error)
            if rpc_error.code() == grpc.StatusCode.NOT_FOUND:
                raise EnvoyNotFoundError
        else:
            health_check_period = response.health_check_period.seconds

            return health_check_period


class RuntimeDirectorClient:
    """
    RuntimeDirectorClient class for experiment manager.

    This class communicates with the director to manage the experiment manager’s
    participation in the federation.

    Attributes:
        stub (director_pb2_grpc.DirectorStub): The gRPC stub for communication
            with the director.
    """

    def __init__(
        self,
        *,
        director_host: str,
        director_port: int,
        tls: bool = False,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
    ) -> None:
        """
        Initialize RuntimeDirectorClient object.

        Args:
            director_host (str): The host name for Director server.
            director_port (int): The port number for Director server.
            tls (bool): Whether to use TLS for the connection.
            root_certificate (Optional[Union[Path, str]]): The path to the root certificate for the
                TLS connection.
            private_key (Optional[Union[Path, str]]): The path to the private key for the TLS
                connection.
            certificate (Optional[Union[Path, str]]): The path to the certificate for the TLS
                connection.

        """
        director_addr = f"{director_host}:{director_port}"
        logger.info("Director address: %s", director_addr)
        if not tls:
            channel = grpc.insecure_channel(director_addr, options=channel_options)
        else:
            if not (root_certificate and private_key and certificate):
                raise Exception("No certificates provided for TLS connection")
            try:
                with open(root_certificate, "rb") as f:
                    root_certificate_b = f.read()
                with open(private_key, "rb") as f:
                    private_key_b = f.read()
                with open(certificate, "rb") as f:
                    certificate_b = f.read()
            except FileNotFoundError as exc:
                raise Exception(f"Provided certificate file is not exist: {exc.filename}")

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificate_b,
                private_key=private_key_b,
                certificate_chain=certificate_b,
            )
            channel = grpc.secure_channel(director_addr, credentials, options=channel_options)
        self.stub = director_pb2_grpc.DirectorStub(channel)

    def set_new_experiment(
        self, experiment_name, col_names, archive_path
    ) -> director_pb2.SetNewExperimentResponse:
        """
        Send the new experiment to director to launch.

        Args:
            experiment_name (str): The name of the experiment.
            col_names (List[str]): The names of the collaborators.
            archive_path (str): The path to the architecture.

        Returns:
            resp (director_pb2.SetNewExperimentResponse): The response from
                the director.
        """
        logger.info("Submitting new experiment %s to director", experiment_name)

        experiment_info_gen = self._get_experiment_info(
            arch_path=archive_path,
            name=experiment_name,
            col_names=col_names,
        )
        resp = self.stub.SetNewExperiment(experiment_info_gen)
        return resp

    def _get_experiment_info(
        self, arch_path, name, col_names
    ) -> Iterator[director_pb2.ExperimentInfo]:
        """
        Generate the experiment data request.

        This method generates a stream of experiment data to be sent to the
        director.

        Args:
            arch_path (str): The path to the architecture.
            name (str): The name of the experiment.
            col_names (List[str]): The names of the collaborators.

        Yields:
            director_pb2.ExperimentInfo: The experiment data.
        """
        with open(arch_path, "rb") as arch:
            max_buffer_size = 2 * 1024 * 1024
            chunk = arch.read(max_buffer_size)
            while chunk != b"":
                if not chunk:
                    raise StopIteration
                experiment_info = director_pb2.ExperimentInfo(
                    name=name,
                    collaborator_names=col_names,
                )
                experiment_info.experiment_data.size = len(chunk)
                experiment_info.experiment_data.exp_data = chunk
                yield experiment_info
                chunk = arch.read(max_buffer_size)

    def get_envoys(self) -> director_pb2.GetEnvoysRequest:
        """Display envoys info in a tabular format.

        Returns:
            envoys (director_pb2.GetEnvoysResponse): The envoy status response
                from the gRPC server.
        """
        envoys = self.stub.GetEnvoys(director_pb2.GetEnvoysRequest())
        return envoys

    def get_flow_state(self) -> Tuple:
        """
        Gets updated state of the flow

        Returns:
            tuple: A tuple containing:
                - completed (bool): Indicates whether the flow has completed.
                - flspec_obj (object): The FLSpec object containing
                    details of the updated flow state.
        """
        response_stream = self.stub.GetFlowState(director_pb2.GetFlowStateRequest())
        response = datastream_to_proto(director_pb2.GetFlowStateResponse(), response_stream)
        return response.completed, response.flspec_obj

    def stream_experiment_stdout(self, experiment_name) -> Iterator[Dict[str, Any]]:
        """Stream experiment stdout RPC.
        Args:
            experiment_name (str): The name of the experiment.
        Yields:
            Dict[str, Any]: The stdout.
        """
        request = director_pb2.GetExperimentStdoutRequest(experiment_name=experiment_name)
        for stdout_message in self.stub.GetExperimentStdout(request):
            yield {
                "stdout_origin": stdout_message.stdout_origin,
                "task_name": stdout_message.task_name,
                "stdout_value": stdout_message.stdout_value,
            }
