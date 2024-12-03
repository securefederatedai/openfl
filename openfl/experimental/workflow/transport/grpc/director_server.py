# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DirectorGRPCServer module."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import grpc
from grpc import aio, ssl_server_credentials

from openfl.experimental.workflow.protocols import director_pb2, director_pb2_grpc
from openfl.experimental.workflow.transport.grpc.exceptions import EnvoyNotFoundError
from openfl.experimental.workflow.transport.grpc.grpc_channel_options import channel_options
from openfl.protocols.utils import get_headers

logger = logging.getLogger(__name__)

CLIENT_ID_DEFAULT = "__default__"


class DirectorGRPCServer(director_pb2_grpc.DirectorServicer):
    """
    Director transport class.

    This class implements a gRPC server for the Director, allowing it to
    communicate with envoys.

    Attributes:
        listen_uri (str): The URI that the server is serving on.
        tls (bool): Whether to use TLS for the connection.
        root_certificate (Optional[Union[Path, str]]): The path to the root certificate for the TLS
            connection.
        private_key (Optional[Union[Path, str]]): The path to the server's private key for the TLS
            connection.
        certificate (Optional[Union[Path, str]]): The path to the server's certificate for the TLS
            connection.
        server (grpc.Server): The gRPC server.
        root_dir (Path): Path to the root directory
        director (Director): The director that this server is serving.
    """

    def __init__(
        self,
        *,
        director_cls,
        tls: bool = True,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
        listen_host: str = "[::]",
        listen_port: int = 50051,
        envoy_health_check_period: int = 0,
        director_config: Optional[Path] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a DirectorGRPCServer object.

        Args:
            director_cls (Type[Director]): The class of the director.
            tls (bool, optional): Whether to use TLS for the connection.
                Defaults to True.
            root_certificate (Optional[Union[Path, str]]): The path
                to the root certificate for the TLS connection. Defaults to
                None.
            private_key (Optional[Union[Path, str]]): The path to
                the server's private key for the TLS connection. Defaults to
                None.
            certificate (Optional[Union[Path, str]]): The path to
                the server's certificate for the TLS connection. Defaults to
                None.
            listen_host (str, optional): The host to listen on. Defaults to
                '[::]'.
            listen_port (int, optional): The port to listen on. Defaults to
                50051.
            director_config (Optional[Path]): Path to director_config file
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.listen_uri = f"{listen_host}:{listen_port}"
        self.tls = tls
        self._fill_certs(root_certificate, private_key, certificate)
        self.server = None
        self.root_dir = Path.cwd()
        self.director = director_cls(
            tls=self.tls,
            root_certificate=self.root_certificate,
            private_key=self.private_key,
            certificate=self.certificate,
            envoy_health_check_period=envoy_health_check_period,
            director_config=director_config,
            **kwargs,
        )

    def _fill_certs(self, root_certificate, private_key, certificate) -> None:
        """Fill certificates.

        Args:
            root_certificate (Union[Path, str]): The path to the root
                certificate for the TLS connection.
            private_key (Union[Path, str]): The path to the server's private
                key for the TLS connection.
            certificate (Union[Path, str]): The path to the server's
                certificate for the TLS connection.
        """
        if self.tls:
            if not all([root_certificate, private_key, certificate]):
                raise ValueError("Incomplete certificates provided")

            self.root_certificate = Path(root_certificate).absolute()
            self.private_key = Path(private_key).absolute()
            self.certificate = Path(certificate).absolute()
        else:
            self.root_certificate = self.private_key = self.certificate = None

    def start(self) -> None:
        """Launch the DirectorGRPCServer"""
        loop = asyncio.get_event_loop()
        loop.create_task(self.director.start_experiment_execution_loop())
        loop.run_until_complete(self._run_server())

    async def _run_server(self) -> None:
        """Run the gRPC server."""
        self.server = aio.server(options=channel_options)
        director_pb2_grpc.add_DirectorServicer_to_server(self, self.server)

        if not self.tls:
            self.server.add_insecure_port(self.listen_uri)
        else:
            with open(self.private_key, "rb") as f:
                private_key_b = f.read()
            with open(self.certificate, "rb") as f:
                certificate_b = f.read()
            with open(self.root_certificate, "rb") as f:
                root_certificate_b = f.read()
            server_credentials = ssl_server_credentials(
                ((private_key_b, certificate_b),),
                root_certificates=root_certificate_b,
                require_client_auth=True,
            )
            self.server.add_secure_port(self.listen_uri, server_credentials)
        logger.info(f"Starting director server on {self.listen_uri}")
        await self.server.start()
        await self.server.wait_for_termination()

    def get_caller(self, context) -> str:
        """Get caller name from context.

        if tls == True: get caller name from auth_context
        if tls == False: get caller name from context header 'client_id'

        Args:
            context (grpc.ServicerContext): The context of the request.

        Returns:
            str: The name of the caller.
        """
        if self.tls:
            return context.auth_context()["x509_common_name"][0].decode("utf-8")
        headers = get_headers(context)
        client_id = headers.get("client_id", CLIENT_ID_DEFAULT)
        return client_id

    def EnvoyConnectionRequest(self, request, context) -> director_pb2.RequestAccepted:
        """Handles a connection request from an Envoy.

        Args:
            request (director_pb2.ConnectEnvoyRequest): The request from
                the envoy
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.RequestAccepted: Indicating if connection was accepted
        """
        logger.info(f"Envoy {request.envoy_name} is attempting to connect")
        is_accepted = self.director.ack_envoy_connection_request(request.envoy_name)
        if is_accepted:
            logger.info(f"Envoy {request.envoy_name} is connected")

        return director_pb2.RequestAccepted(accepted=is_accepted)

    async def UpdateEnvoyStatus(self, request, context) -> director_pb2.UpdateEnvoyStatusResponse:
        """Accept health check from envoy.

        Args:
            request (director_pb2.UpdateEnvoyStatusRequest): The request from
                the envoy.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            resp (director_pb2.UpdateEnvoyStatusResponse): The response to the
                request.
        """
        logger.debug("Updating envoy status: %s", request)
        try:
            health_check_period = self.director.update_envoy_status(
                envoy_name=request.name,
                is_experiment_running=request.is_experiment_running,
            )
        except EnvoyNotFoundError as exc:
            logger.error(exc)
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        else:
            resp = director_pb2.UpdateEnvoyStatusResponse()
            resp.health_check_period.seconds = health_check_period

            return resp

    async def GetEnvoys(self, request, context) -> director_pb2.GetEnvoysResponse:
        """Get status of connected envoys.

        Args:
            request (director_pb2.GetEnvoysRequest): The request from
                the envoy.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetEnvoysResponse: The response to the request.
        """
        envoy_infos = self.director.get_envoys()
        envoy_statuses = []
        for envoy_name, envoy_info in envoy_infos.items():
            envoy_info_message = director_pb2.EnvoyInfo(
                envoy_name=envoy_name,
                is_online=envoy_info["is_online"],
                is_experiment_running=envoy_info["is_experiment_running"],
                experiment_name=envoy_info["experiment_name"],
            )
            envoy_info_message.valid_duration.seconds = envoy_info["valid_duration"]
            envoy_info_message.last_updated.seconds = int(envoy_info["last_updated"])

            envoy_statuses.append(envoy_info_message)

        return director_pb2.GetEnvoysResponse(envoy_infos=envoy_statuses)

    async def GetExperimentData(
        self, request, context
    ) -> AsyncIterator[director_pb2.ExperimentData]:
        """Receive experiment data.

        Args:
            request (director_pb2.GetExperimentDataRequest): The request from
                the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Yields:
            director_pb2.ExperimentData: The experiment data.
        """
        data_file_path = self.director.get_experiment_data(request.experiment_name)
        max_buffer_size = 2 * 1024 * 1024
        with open(data_file_path, "rb") as df:
            while True:
                data = df.read(max_buffer_size)
                if len(data) == 0:
                    break
                yield director_pb2.ExperimentData(size=len(data), exp_data=data)

    async def WaitExperiment(self, request, context) -> director_pb2.WaitExperimentResponse:
        """Handles a request to wait for an experiment to be ready.

        Args:
            request (director_pb2.WaitExperimentRequest): The request from the
                collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.WaitExperimentResponse: The response to the request.
        """
        logger.debug(
            "Request WaitExperiment received from envoy %s",
            request.collaborator_name,
        )
        experiment_name = await self.director.wait_experiment(request.collaborator_name)
        logger.debug(
            "Experiment %s is ready for %s",
            experiment_name,
            request.collaborator_name,
        )

        return director_pb2.WaitExperimentResponse(experiment_name=experiment_name)

    async def SetNewExperiment(self, stream, context) -> director_pb2.SetNewExperimentResponse:
        """Request to set new experiment.

        Args:
            stream (grpc.aio._MultiThreadedRendezvous): The stream of
                experiment data.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.SetNewExperimentResponse: The response to the request.
        """
        data_file_path = self.root_dir / str(uuid.uuid4())
        with open(data_file_path, "wb") as data_file:
            async for request in stream:
                if request.experiment_data.size == len(request.experiment_data.exp_data):
                    data_file.write(request.experiment_data.exp_data)
                else:
                    raise Exception("Could not register new experiment")

        caller = self.get_caller(context)

        is_accepted = await self.director.set_new_experiment(
            experiment_name=request.name,
            sender_name=caller,
            collaborator_names=request.collaborator_names,
            experiment_archive_path=data_file_path,
        )

        logger.info("Experiment %s registered", request.name)
        return director_pb2.SetNewExperimentResponse(status=is_accepted)

    async def GetFlowState(self, request, context) -> director_pb2.GetFlowStateResponse:
        """Get updated flow after experiment is finished.

        Args:
            request (director_pb2.GetFlowStatusRequest): The request from
                the experiment manager
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetFlowStateResponse: The response to the request.
        """
        status, flspec_obj = await self.director.get_flow_state()
        return director_pb2.GetFlowStateResponse(completed=status, flspec_obj=flspec_obj)

    async def GetExperimentStdout(
        self, request, context
    ) -> AsyncIterator[director_pb2.GetExperimentStdoutResponse]:
        """
        Request to stream stdout from the aggregator to frontend.

        Args:
            request (director_pb2.GetExperimentStdoutRequest): The request from
                the experiment manager.
            context (grpc.ServicerContext): The context of the request.

        Yields:
            director_pb2.GetExperimentStdoutResponse: The metrics.
        """
        logger.info("Getting standard output for experiment: %s...", request.experiment_name)
        caller = self.get_caller(context)
        async for stdout_dict in self.director.stream_experiment_stdout(
            experiment_name=request.experiment_name, caller=caller
        ):
            if stdout_dict is None:
                await asyncio.sleep(1)
                continue
            yield director_pb2.GetExperimentStdoutResponse(**stdout_dict)
