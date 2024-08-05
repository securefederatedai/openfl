# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Director clients module."""

import logging
from datetime import datetime
from typing import List, Type

import grpc

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2, director_pb2_grpc, interceptors
from openfl.protocols.utils import construct_model_proto, deconstruct_model_proto
from openfl.transport.grpc.director_server import CLIENT_ID_DEFAULT
from openfl.transport.grpc.exceptions import ShardNotFoundError
from openfl.transport.grpc.grpc_channel_options import channel_options

logger = logging.getLogger(__name__)


class ShardDirectorClient:
    """
    The internal director client class.

    This class communicates with the director to manage the shard's
    participation in the federation.

    Attributes:
        shard_name (str): The name of the shard.
        stub (director_pb2_grpc.DirectorStub): The gRPC stub for communication
            with the director.
    """

    def __init__(
        self,
        *,
        director_host,
        director_port,
        shard_name,
        tls=True,
        root_certificate=None,
        private_key=None,
        certificate=None,
    ) -> None:
        """
        Initialize a shard director client object.

        Args:
            director_host (str): The host of the director.
            director_port (int): The port of the director.
            shard_name (str): The name of the shard.
            tls (bool): Whether to use TLS for the connection.
            root_certificate (str): The path to the root certificate for the
                TLS connection.
            private_key (str): The path to the private key for the TLS
                connection.
            certificate (str): The path to the certificate for the TLS
                connection.
        """
        self.shard_name = shard_name
        director_addr = f"{director_host}:{director_port}"
        logger.info("Director address: %s", director_addr)
        if not tls:
            channel = grpc.insecure_channel(director_addr, options=channel_options)
        else:
            if not (root_certificate and private_key and certificate):
                raise Exception("No certificates provided")
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

    def report_shard_info(
        self, shard_descriptor: Type[ShardDescriptor], cuda_devices: tuple
    ) -> bool:
        """
        Report shard info to the director.

        Args:
            shard_descriptor (Type[ShardDescriptor]): The descriptor of the
                shard.
            cuda_devices (tuple): The CUDA devices available on the shard.

        Returns:
            acknowledgement (bool): Whether the report was accepted by the
                director.
        """
        logger.info("Sending %s shard info to director", self.shard_name)
        # True considered as successful registration
        shard_info = director_pb2.ShardInfo(
            shard_description=shard_descriptor.dataset_description,
            sample_shape=shard_descriptor.sample_shape,
            target_shape=shard_descriptor.target_shape,
        )

        shard_info.node_info.name = self.shard_name
        shard_info.node_info.cuda_devices.extend(
            director_pb2.CudaDeviceInfo(index=cuda_device) for cuda_device in cuda_devices
        )

        request = director_pb2.UpdateShardInfoRequest(shard_info=shard_info)
        acknowledgement = self.stub.UpdateShardInfo(request)
        return acknowledgement.accepted

    def wait_experiment(self):
        """
        Wait an experiment data from the director.

        Returns:
            experiment_name (str): The name of the experiment.
        """
        logger.info("Waiting for an experiment to run...")
        response = self.stub.WaitExperiment(self._get_experiment_data())
        logger.info("New experiment received: %s", response)
        experiment_name = response.experiment_name
        if not experiment_name:
            raise Exception("No experiment")

        return experiment_name

    def get_experiment_data(self, experiment_name):
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
            experiment_name=experiment_name, collaborator_name=self.shard_name
        )
        data_stream = self.stub.GetExperimentData(request)

        return data_stream

    def set_experiment_failed(
        self,
        experiment_name: str,
        error_code: int = 1,
        error_description: str = "",
    ):
        """
        Set the experiment failed.

        Args:
            experiment_name (str): The name of the experiment.
            error_code (int, optional): The error code. Defaults to 1.
            error_description (str, optional): The description of the error.
                Defaults to ''.
        """
        logger.info("Experiment %s failed", experiment_name)
        request = director_pb2.SetExperimentFailedRequest(
            experiment_name=experiment_name,
            collaborator_name=self.shard_name,
            error_code=error_code,
            error_description=error_description,
        )
        self.stub.SetExperimentFailed(request)

    def _get_experiment_data(self):
        """Generate the experiment data request.

        Returns:
            director_pb2.WaitExperimentRequest: The request for experiment
                data.
        """
        return director_pb2.WaitExperimentRequest(collaborator_name=self.shard_name)

    def send_health_check(
        self,
        *,
        envoy_name: str,
        is_experiment_running: bool,
        cuda_devices_info: List[dict] = None,
    ) -> int:
        """Send envoy health check.

        Args:
            envoy_name (str): The name of the envoy.
            is_experiment_running (bool): Whether an experiment is currently
                running.
            cuda_devices_info (List[dict], optional): Information about the
                CUDA devices. Defaults to None.

        Returns:
            health_check_period (int): The period for health checks.
        """
        status = director_pb2.UpdateEnvoyStatusRequest(
            name=envoy_name,
            is_experiment_running=is_experiment_running,
        )

        cuda_messages = []
        if cuda_devices_info is not None:
            try:
                cuda_messages = [director_pb2.CudaDeviceInfo(**item) for item in cuda_devices_info]
            except Exception as e:
                logger.info("%s", e)

        status.cuda_devices.extend(cuda_messages)

        logger.debug("Sending health check status: %s", status)

        try:
            response = self.stub.UpdateEnvoyStatus(status)
        except grpc.RpcError as rpc_error:
            logger.error(rpc_error)
            if rpc_error.code() == grpc.StatusCode.NOT_FOUND:
                raise ShardNotFoundError
        else:
            health_check_period = response.health_check_period.seconds

            return health_check_period


class DirectorClient:
    """Director client class for users.

    This class communicates with the director to manage the user's
    participation in the federation.

    Attributes:
        stub (director_pb2_grpc.DirectorStub): The gRPC stub for communication
            with the director.
    """

    def __init__(
        self,
        *,
        client_id: str,
        director_host: str,
        director_port: int,
        tls: bool,
        root_certificate: str,
        private_key: str,
        certificate: str,
    ) -> None:
        """
        Initialize director client object.

        Args:
            client_id (str): The ID of the client.
            director_host (str): The host of the director.
            director_port (int): The port of the director.
            tls (bool): Whether to use TLS for the connection.
            root_certificate (str): The path to the root certificate for the
                TLS connection.
            private_key (str): The path to the private key for the TLS
                connection.
            certificate (str): The path to the certificate for the TLS
                connection.
        """
        director_addr = f"{director_host}:{director_port}"
        if not tls:
            if not client_id:
                client_id = CLIENT_ID_DEFAULT
            channel = grpc.insecure_channel(director_addr, options=channel_options)
            headers = {
                "client_id": client_id,
            }
            header_interceptor = interceptors.headers_adder(headers)
            channel = grpc.intercept_channel(channel, header_interceptor)
        else:
            if not (root_certificate and private_key and certificate):
                raise Exception("No certificates provided")
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

    def set_new_experiment(self, name, col_names, arch_path, initial_tensor_dict=None):
        """
        Send the new experiment to director to launch.

        Args:
            name (str): The name of the experiment.
            col_names (List[str]): The names of the collaborators.
            arch_path (str): The path to the architecture.
            initial_tensor_dict (dict, optional): The initial tensor
                dictionary. Defaults to None.

        Returns:
            resp (director_pb2.SetNewExperimentResponse): The response from
                the director.
        """
        logger.info("Submitting new experiment %s to director", name)
        if initial_tensor_dict:
            model_proto = construct_model_proto(initial_tensor_dict, 0, NoCompressionPipeline())
            experiment_info_gen = self._get_experiment_info(
                arch_path=arch_path,
                name=name,
                col_names=col_names,
                model_proto=model_proto,
            )
            resp = self.stub.SetNewExperiment(experiment_info_gen)
            return resp

    def _get_experiment_info(self, arch_path, name, col_names, model_proto):
        """
        Generate the experiment data request.

        This method generates a stream of experiment data to be sent to the
        director.

        Args:
            arch_path (str): The path to the architecture.
            name (str): The name of the experiment.
            col_names (List[str]): The names of the collaborators.
            model_proto (ModelProto): The initial model.

        Yields:
            director_pb2.ExperimentInfo: The experiment data.
        """
        with open(arch_path, "rb") as arch:
            max_buffer_size = 2 * 1024 * 1024
            chunk = arch.read(max_buffer_size)
            while chunk != b"":
                if not chunk:
                    raise StopIteration
                # TODO: add hash or/and size to check
                experiment_info = director_pb2.ExperimentInfo(
                    name=name,
                    collaborator_names=col_names,
                    model_proto=model_proto,
                )
                experiment_info.experiment_data.size = len(chunk)
                experiment_info.experiment_data.npbytes = chunk
                yield experiment_info
                chunk = arch.read(max_buffer_size)

    def get_experiment_status(self, experiment_name):
        """
        Check if the experiment was accepted by the director.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            resp (director_pb2.GetExperimentStatusResponse): The response from
                the director.
        """
        logger.info("Getting experiment Status...")
        request = director_pb2.GetExperimentStatusRequest(experiment_name=experiment_name)
        resp = self.stub.GetExperimentStatus(request)
        return resp

    def get_dataset_info(self):
        """Request the dataset info from the director.

        Returns:
            Tuple[List[int], List[int]]: The sample shape and target shape of
                the dataset.
        """
        resp = self.stub.GetDatasetInfo(director_pb2.GetDatasetInfoRequest())
        return resp.shard_info.sample_shape, resp.shard_info.target_shape

    def _get_trained_model(self, experiment_name, model_type):
        """Get trained model RPC.

        Args:
            experiment_name (str): The name of the experiment.
            model_type (director_pb2.GetTrainedModelRequest.ModelType): The
                type of the model.

        Returns:
            tensor_dict (Dict[str, numpy.ndarray]): The trained model.
        """
        get_model_request = director_pb2.GetTrainedModelRequest(
            experiment_name=experiment_name,
            model_type=model_type,
        )
        model_proto_response = self.stub.GetTrainedModel(get_model_request)
        tensor_dict, _ = deconstruct_model_proto(
            model_proto_response.model_proto,
            NoCompressionPipeline(),
        )
        return tensor_dict

    def get_best_model(self, experiment_name):
        """Get best model method.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            Dict[str, numpy.ndarray]: The best model.
        """
        model_type = director_pb2.GetTrainedModelRequest.BEST_MODEL
        return self._get_trained_model(experiment_name, model_type)

    def get_last_model(self, experiment_name):
        """Get last model method.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            Dict[str, numpy.ndarray]: The last model.
        """
        model_type = director_pb2.GetTrainedModelRequest.LAST_MODEL
        return self._get_trained_model(experiment_name, model_type)

    def stream_metrics(self, experiment_name):
        """Stream metrics RPC.

        Args:
            experiment_name (str): The name of the experiment.

        Yields:
            Dict[str, Any]: The metrics.
        """
        request = director_pb2.GetMetricStreamRequest(experiment_name=experiment_name)
        for metric_message in self.stub.GetMetricStream(request):
            yield {
                "metric_origin": metric_message.metric_origin,
                "task_name": metric_message.task_name,
                "metric_name": metric_message.metric_name,
                "metric_value": metric_message.metric_value,
                "round": metric_message.round,
            }

    def remove_experiment_data(self, name):
        """Remove experiment data RPC.

        Args:
            name (str): The name of the experiment.

        Returns:
            bool: Whether the removal was acknowledged.
        """
        request = director_pb2.RemoveExperimentRequest(experiment_name=name)
        response = self.stub.RemoveExperimentData(request)
        return response.acknowledgement

    def get_envoys(self, raw_result=False):
        """Get envoys info.

        Args:
            raw_result (bool, optional): Whether to return the raw result.
                Defaults to False.

        Returns:
            result (Union[director_pb2.GetEnvoysResponse,
                Dict[str, Dict[str, Any]]]): The envoys info.
        """
        envoys = self.stub.GetEnvoys(director_pb2.GetEnvoysRequest())
        if raw_result:
            return envoys
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {}
        for envoy in envoys.envoy_infos:
            result[envoy.shard_info.node_info.name] = {
                "shard_info": envoy.shard_info,
                "is_online": envoy.is_online or False,
                "is_experiment_running": envoy.is_experiment_running or False,
                "last_updated": datetime.fromtimestamp(envoy.last_updated.seconds).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "current_time": now,
                "valid_duration": envoy.valid_duration,
                "experiment_name": "ExperimentName Mock",
            }
        return result

    def get_experiments_list(self):
        """
        Get experiments list.

        Returns:
            List[str]: The list of experiments.
        """
        response = self.stub.GetExperimentsList(director_pb2.GetExperimentsListRequest())
        return response.experiments

    def get_experiment_description(self, name):
        """Get experiment info.

        Args:
            name (str): The name of the experiment.

        Returns:
            director_pb2.ExperimentDescription: The description of the
                experiment.
        """
        response = self.stub.GetExperimentDescription(
            director_pb2.GetExperimentDescriptionRequest(name=name)
        )
        return response.experiment
