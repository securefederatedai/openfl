# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Director server."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional, Union

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict
from grpc import aio, ssl_server_credentials

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import base_pb2, director_pb2, director_pb2_grpc
from openfl.protocols.utils import construct_model_proto, deconstruct_model_proto, get_headers
from openfl.transport.grpc.exceptions import ShardNotFoundError
from openfl.transport.grpc.grpc_channel_options import channel_options

logger = logging.getLogger(__name__)

CLIENT_ID_DEFAULT = "__default__"


class DirectorGRPCServer(director_pb2_grpc.DirectorServicer):
    """
    Director transport class.

    This class implements a gRPC server for the Director, allowing it to
    communicate with collaborators.

    Attributes:
        director (Director): The director that this server is serving.
        listen_uri (str): The URI that the server is serving on.
        tls (bool): Whether to use TLS for the connection.
        root_certificate (Path): The path to the root certificate for the TLS
            connection.
        private_key (Path): The path to the server's private key for the TLS
            connection.
        certificate (Path): The path to the server's certificate for the TLS
            connection.
        server (grpc.Server): The gRPC server.
    """

    def __init__(
        self,
        *,
        director_cls,
        tls: bool = True,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
        review_plan_callback: Union[None, Callable] = None,
        listen_host: str = "[::]",
        listen_port: int = 50051,
        envoy_health_check_period: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialize a director object.

        Args:
            director_cls (Type[Director]): The class of the director.
            tls (bool, optional): Whether to use TLS for the connection.
                Defaults to True.
            root_certificate (Optional[Union[Path, str]], optional): The path
                to the root certificate for the TLS connection. Defaults to
                None.
            private_key (Optional[Union[Path, str]], optional): The path to
                the server's private key for the TLS connection. Defaults to
                None.
            certificate (Optional[Union[Path, str]], optional): The path to
                the server's certificate for the TLS connection. Defaults to
                None.
            review_plan_callback (Union[None, Callable], optional): The
                callback for reviewing the plan. Defaults to None.
            listen_host (str, optional): The host to listen on. Defaults to
                '[::]'.
            listen_port (int, optional): The port to listen on. Defaults to
                50051.
            envoy_health_check_period (int, optional): The period for health
                checks. Defaults to 0.
            **kwargs: Additional keyword arguments.
        """
        # TODO: add working directory
        super().__init__()

        self.listen_uri = f"{listen_host}:{listen_port}"
        self.tls = tls
        self.root_certificate = None
        self.private_key = None
        self.certificate = None
        self._fill_certs(root_certificate, private_key, certificate)
        self.server = None
        self.root_dir = Path.cwd()
        self.director = director_cls(
            tls=self.tls,
            root_certificate=self.root_certificate,
            private_key=self.private_key,
            certificate=self.certificate,
            review_plan_callback=review_plan_callback,
            envoy_health_check_period=envoy_health_check_period,
            **kwargs,
        )

    def _fill_certs(self, root_certificate, private_key, certificate):
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
            if not (root_certificate and private_key and certificate):
                raise Exception("No certificates provided")
            self.root_certificate = Path(root_certificate).absolute()
            self.private_key = Path(private_key).absolute()
            self.certificate = Path(certificate).absolute()

    def get_caller(self, context):
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

    def start(self):
        """Launch the director GRPC server."""
        loop = asyncio.get_event_loop()  # TODO: refactor after end of support for python3.6
        loop.create_task(self.director.start_experiment_execution_loop())
        loop.run_until_complete(self._run_server())

    async def _run_server(self):
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
        logger.info("Starting director server on %s", self.listen_uri)
        await self.server.start()
        await self.server.wait_for_termination()

    async def UpdateShardInfo(self, request, context):  # NOQA:N802
        """
        Receive acknowledge shard info.

        Args:
            request (director_pb2.UpdateShardInfoRequest): The request from
                the shard.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            reply (director_pb2.UpdateShardInfoResponse): The response to the
                request.
        """
        logger.info("Updating shard info: %s", request.shard_info)
        dict_shard_info = MessageToDict(request.shard_info, preserving_proto_field_name=True)
        is_accepted = self.director.acknowledge_shard(dict_shard_info)
        reply = director_pb2.UpdateShardInfoResponse(accepted=is_accepted)

        return reply

    async def SetNewExperiment(self, stream, context):  # NOQA:N802
        """Request to set new experiment.

        Args:
            stream (grpc.aio._MultiThreadedRendezvous): The stream of
                experiment data.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.SetNewExperimentResponse: The response to the request.
        """
        # TODO: add streaming reader
        data_file_path = self.root_dir / str(uuid.uuid4())
        with open(data_file_path, "wb") as data_file:
            async for request in stream:
                if request.experiment_data.size == len(request.experiment_data.npbytes):
                    data_file.write(request.experiment_data.npbytes)
                else:
                    raise Exception("Could not register new experiment")

        tensor_dict = None
        if request.model_proto:
            tensor_dict, _ = deconstruct_model_proto(request.model_proto, NoCompressionPipeline())

        caller = self.get_caller(context)

        is_accepted = await self.director.set_new_experiment(
            experiment_name=request.name,
            sender_name=caller,
            tensor_dict=tensor_dict,
            collaborator_names=request.collaborator_names,
            experiment_archive_path=data_file_path,
        )

        logger.info("Experiment %s registered", request.name)
        return director_pb2.SetNewExperimentResponse(accepted=is_accepted)

    async def GetExperimentStatus(self, request, context):  # NOQA: N802
        """
        Get experiment status and update if experiment was approved.

        Args:
            request (director_pb2.GetExperimentStatusRequest): The request
                from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetExperimentStatusResponse: The response to the
                request.
        """
        logger.debug("GetExperimentStatus request received")
        caller = self.get_caller(context)
        experiment_status = await self.director.get_experiment_status(
            experiment_name=request.experiment_name, caller=caller
        )
        logger.debug("Sending GetExperimentStatus response")
        return director_pb2.GetExperimentStatusResponse(experiment_status=experiment_status)

    async def GetTrainedModel(self, request, context):  # NOQA:N802
        """
        RPC for retrieving trained models.

        Args:
            request (director_pb2.GetTrainedModelRequest): The request from
                the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.TrainedModelResponse: The response to the request.
        """
        logger.debug("Received request for trained model...")

        if request.model_type == director_pb2.GetTrainedModelRequest.BEST_MODEL:
            model_type = "best"
        elif request.model_type == director_pb2.GetTrainedModelRequest.LAST_MODEL:
            model_type = "last"
        else:
            logger.error("Incorrect model type")
            return director_pb2.TrainedModelResponse()

        caller = self.get_caller(context)

        trained_model_dict = self.director.get_trained_model(
            experiment_name=request.experiment_name,
            caller=caller,
            model_type=model_type,
        )

        if trained_model_dict is None:
            return director_pb2.TrainedModelResponse()

        model_proto = construct_model_proto(trained_model_dict, 0, NoCompressionPipeline())

        logger.debug("Sending trained model")

        return director_pb2.TrainedModelResponse(model_proto=model_proto)

    async def GetExperimentData(self, request, context):  # NOQA:N802
        """Receive experiment data.

        Args:
            request (director_pb2.GetExperimentDataRequest): The request from
                the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Yields:
            director_pb2.ExperimentData: The experiment data.
        """
        # TODO: add size filling
        # TODO: add experiment name field
        # TODO: rename npbytes to data
        data_file_path = self.director.get_experiment_data(request.experiment_name)
        max_buffer_size = 2 * 1024 * 1024
        with open(data_file_path, "rb") as df:
            while True:
                data = df.read(max_buffer_size)
                if len(data) == 0:
                    break
                yield director_pb2.ExperimentData(size=len(data), npbytes=data)

    async def WaitExperiment(self, request, context):  # NOQA:N802
        """
        Request for wait an experiment.

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

    async def GetDatasetInfo(self, request, context):  # NOQA:N802
        """
        Request the info about target and sample shapes in the dataset.

        Args:
            request (director_pb2.GetDatasetInfoRequest): The request from the
                collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetDatasetInfoResponse: The response to the request.
        """
        logger.debug("Received request for dataset info...")

        sample_shape, target_shape = self.director.get_dataset_info()
        shard_info = director_pb2.ShardInfo(sample_shape=sample_shape, target_shape=target_shape)
        resp = director_pb2.GetDatasetInfoResponse(shard_info=shard_info)

        logger.debug("Sending dataset info")
        return resp

    async def GetMetricStream(self, request, context):  # NOQA:N802
        """
        Request to stream metrics from the aggregator to frontend.

        Args:
            request (director_pb2.GetMetricStreamRequest): The request from
                the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Yields:
            director_pb2.GetMetricStreamResponse: The metrics.
        """
        logger.info("Getting metrics for %s...", request.experiment_name)

        caller = self.get_caller(context)
        async for metric_dict in self.director.stream_metrics(
            experiment_name=request.experiment_name, caller=caller
        ):
            if metric_dict is None:
                await asyncio.sleep(1)
                continue
            yield director_pb2.GetMetricStreamResponse(**metric_dict)

    async def RemoveExperimentData(self, request, context):  # NOQA:N802
        """Remove experiment data RPC.

        Args:
            request (director_pb2.RemoveExperimentRequest): The request from
                the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            response (director_pb2.RemoveExperimentResponse): The response to
                the request.
        """
        response = director_pb2.RemoveExperimentResponse(acknowledgement=False)
        caller = self.get_caller(context)
        self.director.remove_experiment_data(
            experiment_name=request.experiment_name,
            caller=caller,
        )

        response.acknowledgement = True
        return response

    async def SetExperimentFailed(self, request, context):  # NOQA:N802
        """Set the experiment failed.

        Args:
            request (director_pb2.SetExperimentFailedRequest): The request
                from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            response (director_pb2.SetExperimentFailedResponse): The response
                to the request.
        """
        response = director_pb2.SetExperimentFailedResponse()
        if self.get_caller(context) != CLIENT_ID_DEFAULT:
            return response
        logger.error(
            f"Collaborator {request.collaborator_name} failed with error code:"
            f" {request.error_code}, error_description: {request.error_description}"
            f"Stopping experiment."
        )
        self.director.set_experiment_failed(
            experiment_name=request.experiment_name,
            collaborator_name=request.collaborator_name,
        )

        return response

    async def UpdateEnvoyStatus(self, request, context):  # NOQA:N802
        """
        Accept health check from envoy.

        Args:
            request (director_pb2.UpdateEnvoyStatusRequest): The request from
                the envoy.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            resp (director_pb2.UpdateEnvoyStatusResponse): The response to the
                request.
        """
        logger.debug("Updating envoy status: %s", request)
        cuda_devices_info = [
            MessageToDict(message, preserving_proto_field_name=True)
            for message in request.cuda_devices
        ]
        try:
            health_check_period = self.director.update_envoy_status(
                envoy_name=request.name,
                is_experiment_running=request.is_experiment_running,
                cuda_devices_status=cuda_devices_info,
            )
        except ShardNotFoundError as exc:
            logger.error(exc)
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        else:
            resp = director_pb2.UpdateEnvoyStatusResponse()
            resp.health_check_period.seconds = health_check_period

            return resp

    async def GetEnvoys(self, request, context):  # NOQA:N802
        """Get a status information about envoys.

        Args:
            request (director_pb2.GetEnvoysRequest): The request from the
                collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetEnvoysResponse: The response to the request.
        """
        envoy_infos = self.director.get_envoys()
        envoy_statuses = []
        for envoy_info in envoy_infos:
            envoy_info_message = director_pb2.EnvoyInfo(
                shard_info=ParseDict(
                    envoy_info["shard_info"],
                    director_pb2.ShardInfo(),
                    ignore_unknown_fields=True,
                ),
                is_online=envoy_info["is_online"],
                is_experiment_running=envoy_info["is_experiment_running"],
            )
            envoy_info_message.valid_duration.seconds = envoy_info["valid_duration"]
            envoy_info_message.last_updated.seconds = int(envoy_info["last_updated"])

            envoy_statuses.append(envoy_info_message)

        return director_pb2.GetEnvoysResponse(envoy_infos=envoy_statuses)

    async def GetExperimentsList(self, request, context):  # NOQA:N802
        """Get list of experiments description.

        Args:
            request (director_pb2.GetExperimentsListRequest): The request from
                the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetExperimentsListResponse: The response to the
                request.
        """
        caller = self.get_caller(context)
        experiments = self.director.get_experiments_list(caller)
        experiment_list = [director_pb2.ExperimentListItem(**exp) for exp in experiments]
        return director_pb2.GetExperimentsListResponse(experiments=experiment_list)

    async def GetExperimentDescription(self, request, context):  # NOQA:N802
        """Get an experiment description.

        Args:
            request (director_pb2.GetExperimentDescriptionRequest): The
                request from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            director_pb2.GetExperimentDescriptionResponse: The response to the
                request.
        """
        caller = self.get_caller(context)
        experiment = self.director.get_experiment_description(caller, request.name)
        models_statuses = [
            base_pb2.DownloadStatus(name=ms["name"], status=ms["status"])
            for ms in experiment["download_statuses"]["models"]
        ]
        logs_statuses = [
            base_pb2.DownloadStatus(name=ls["name"], status=ls["status"])
            for ls in experiment["download_statuses"]["logs"]
        ]
        download_statuses = base_pb2.DownloadStatuses(
            models=models_statuses,
            logs=logs_statuses,
        )
        collaborators = [
            base_pb2.CollaboratorDescription(
                name=col["name"],
                status=col["status"],
                progress=col["progress"],
                round=col["round"],
                current_task=col["current_task"],
                next_task=col["next_task"],
            )
            for col in experiment["collaborators"]
        ]
        tasks = [
            base_pb2.TaskDescription(name=task["name"], description=task["description"])
            for task in experiment["tasks"]
        ]

        return director_pb2.GetExperimentDescriptionResponse(
            experiment=base_pb2.ExperimentDescription(
                name=experiment["name"],
                status=experiment["status"],
                progress=experiment["progress"],
                current_round=experiment["current_round"],
                total_rounds=experiment["total_rounds"],
                download_statuses=download_statuses,
                collaborators=collaborators,
                tasks=tasks,
            ),
        )
