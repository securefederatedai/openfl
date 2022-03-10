# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director clients module."""

import logging
from datetime import datetime
from typing import List
from typing import Type

import grpc

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.protocols import interceptors
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto
from openfl.transport.grpc.director_server import CLIENT_ID_DEFAULT

logger = logging.getLogger(__name__)


class ShardDirectorClient:
    """The internal director client class."""

    def __init__(self, *, director_host, director_port, shard_name, tls=True,
                 root_certificate=None, private_key=None, certificate=None) -> None:
        """Initialize a shard director client object."""
        self.shard_name = shard_name
        director_addr = f'{director_host}:{director_port}'
        options = [('grpc.max_message_length', 100 * 1024 * 1024)]
        if not tls:
            channel = grpc.insecure_channel(director_addr, options=options)
        else:
            if not (root_certificate and private_key and certificate):
                raise Exception('No certificates provided')
            try:
                with open(root_certificate, 'rb') as f:
                    root_certificate_b = f.read()
                with open(private_key, 'rb') as f:
                    private_key_b = f.read()
                with open(certificate, 'rb') as f:
                    certificate_b = f.read()
            except FileNotFoundError as exc:
                raise Exception(f'Provided certificate file is not exist: {exc.filename}')

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificate_b,
                private_key=private_key_b,
                certificate_chain=certificate_b
            )
            channel = grpc.secure_channel(director_addr, credentials, options=options)
        self.stub = director_pb2_grpc.DirectorStub(channel)

    def report_shard_info(self, shard_descriptor: Type[ShardDescriptor],
                          cuda_devices: tuple) -> bool:
        """Report shard info to the director."""
        logger.info('Send report UpdateShardInfo')
        # True considered as successful registration
        shard_info = director_pb2.ShardInfo(
            shard_description=shard_descriptor.dataset_description,
            sample_shape=shard_descriptor.sample_shape,
            target_shape=shard_descriptor.target_shape
        )

        shard_info.node_info.name = self.shard_name
        shard_info.node_info.cuda_devices.extend(
            director_pb2.CudaDeviceInfo(index=cuda_device)
            for cuda_device in cuda_devices
        )

        request = director_pb2.UpdateShardInfoRequest(shard_info=shard_info)
        acknowledgement = self.stub.UpdateShardInfo(request)
        return acknowledgement.accepted

    def wait_experiment(self):
        """Wait an experiment data from the director."""
        logger.info('Send WaitExperiment request')
        response_iter = self.stub.WaitExperiment(self._get_experiment_data())
        logger.info('WaitExperiment response has received')
        response = next(response_iter)
        experiment_name = response.experiment_name
        if not experiment_name:
            raise Exception('No experiment')

        return experiment_name

    def get_experiment_data(self, experiment_name):
        """Get an experiment data from the director."""
        logger.info(f'Request experiment {experiment_name}')
        request = director_pb2.GetExperimentDataRequest(
            experiment_name=experiment_name,
            collaborator_name=self.shard_name
        )
        data_stream = self.stub.GetExperimentData(request)

        return data_stream

    def set_experiment_failed(
            self,
            experiment_name: str,
            error_code: int = 1,
            error_description: str = ''
    ):
        """Set the experiment failed."""
        request = director_pb2.SetExperimentFailedRequest(
            experiment_name=experiment_name,
            collaborator_name=self.shard_name,
            error_code=error_code,
            error_description=error_description
        )
        self.stub.SetExperimentFailed(request)

    def _get_experiment_data(self):
        """Generate the experiment data request."""
        yield director_pb2.WaitExperimentRequest(collaborator_name=self.shard_name)

    def send_health_check(
            self, *,
            envoy_name: str,
            is_experiment_running: bool,
            cuda_devices_info: List[dict] = None,
    ) -> int:
        """Send envoy health check."""
        status = director_pb2.UpdateEnvoyStatusRequest(
            name=envoy_name,
            is_experiment_running=is_experiment_running,
        )

        cuda_messages = []
        if cuda_devices_info is not None:
            try:
                cuda_messages = [
                    director_pb2.CudaDeviceInfo(**item)
                    for item in cuda_devices_info
                ]
            except Exception as e:
                logger.info(f'{e}')

        status.cuda_devices.extend(cuda_messages)

        logger.debug(f'Sending health check status: {status}')

        response = self.stub.UpdateEnvoyStatus(status)
        health_check_period = response.health_check_period.seconds

        return health_check_period


class DirectorClient:
    """Director client class for users."""

    def __init__(
            self, *,
            client_id: str,
            director_host: str,
            director_port: int,
            tls: bool,
            root_certificate: str,
            private_key: str,
            certificate: str,
    ) -> None:
        """Initialize director client object."""
        director_addr = f'{director_host}:{director_port}'
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        if not tls:
            if not client_id:
                client_id = CLIENT_ID_DEFAULT
            channel = grpc.insecure_channel(director_addr, options=channel_opt)
            headers = {
                'client_id': client_id,
            }
            header_interceptor = interceptors.headers_adder(headers)
            channel = grpc.intercept_channel(channel, header_interceptor)
        else:
            if not (root_certificate and private_key and certificate):
                raise Exception('No certificates provided')
            try:
                with open(root_certificate, 'rb') as f:
                    root_certificate_b = f.read()
                with open(private_key, 'rb') as f:
                    private_key_b = f.read()
                with open(certificate, 'rb') as f:
                    certificate_b = f.read()
            except FileNotFoundError as exc:
                raise Exception(f'Provided certificate file is not exist: {exc.filename}')

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificate_b,
                private_key=private_key_b,
                certificate_chain=certificate_b
            )

            channel = grpc.secure_channel(director_addr, credentials, options=channel_opt)
        self.stub = director_pb2_grpc.DirectorStub(channel)

    def set_new_experiment(self, name, col_names, arch_path,
                           initial_tensor_dict=None):
        """Send the new experiment to director to launch."""
        logger.info('SetNewExperiment')
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
        with open(arch_path, 'rb') as arch:
            max_buffer_size = 2 * 1024 * 1024
            chunk = arch.read(max_buffer_size)
            while chunk != b'':
                if not chunk:
                    raise StopIteration
                # TODO: add hash or/and size to check
                experiment_info = director_pb2.ExperimentInfo(
                    name=name,
                    collaborator_names=col_names,
                    model_proto=model_proto
                )
                experiment_info.experiment_data.size = len(chunk)
                experiment_info.experiment_data.npbytes = chunk
                yield experiment_info
                chunk = arch.read(max_buffer_size)

    def get_dataset_info(self):
        """Request the dataset info from the director."""
        resp = self.stub.GetDatasetInfo(director_pb2.GetDatasetInfoRequest())
        return resp.shard_info.sample_shape, resp.shard_info.target_shape

    def _get_trained_model(self, experiment_name, model_type):
        """Get trained model RPC."""
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
        """Get best model method."""
        model_type = director_pb2.GetTrainedModelRequest.BEST_MODEL
        return self._get_trained_model(experiment_name, model_type)

    def get_last_model(self, experiment_name):
        """Get last model method."""
        model_type = director_pb2.GetTrainedModelRequest.LAST_MODEL
        return self._get_trained_model(experiment_name, model_type)

    def stream_metrics(self, experiment_name):
        """Stream metrics RPC."""
        request = director_pb2.GetMetricStreamRequest(experiment_name=experiment_name)
        for metric_message in self.stub.GetMetricStream(request):
            yield {
                'metric_origin': metric_message.metric_origin,
                'task_name': metric_message.task_name,
                'metric_name': metric_message.metric_name,
                'metric_value': metric_message.metric_value,
                'round': metric_message.round,
            }

    def remove_experiment_data(self, name):
        """Remove experiment data RPC."""
        request = director_pb2.RemoveExperimentRequest(experiment_name=name)
        response = self.stub.RemoveExperimentData(request)
        return response.acknowledgement

    def get_envoys(self, raw_result=False):
        """Get envoys info."""
        envoys = self.stub.GetEnvoys(director_pb2.GetEnvoysRequest())
        if raw_result:
            return envoys
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = {}
        for envoy in envoys.envoy_infos:
            result[envoy.shard_info.node_info.name] = {
                'shard_info': envoy.shard_info,
                'is_online': envoy.is_online or False,
                'is_experiment_running': envoy.is_experiment_running or False,
                'last_updated': datetime.fromtimestamp(
                    envoy.last_updated.seconds).strftime('%Y-%m-%d %H:%M:%S'),
                'current_time': now,
                'valid_duration': envoy.valid_duration,
                'experiment_name': 'ExperimentName Mock',
            }
        return result

    def get_experiments_list(self):
        """Get experiments list."""
        response = self.stub.GetExperimentsList(
            director_pb2.GetExperimentsListRequest()
        )
        return response.experiments

    def get_experiment_description(self, name):
        """Get experiment info."""
        response = self.stub.GetExperimentDescription(
            director_pb2.GetExperimentDescriptionRequest(name=name)
        )
        return response.experiment
