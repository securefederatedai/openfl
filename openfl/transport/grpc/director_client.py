# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director clients module."""

import logging
from datetime import datetime

import grpc

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto

logger = logging.getLogger(__name__)


class ShardDirectorClient:
    """The internal director client class."""

    def __init__(self, director_uri, shard_name, tls=True,
                 root_ca=None, key=None, cert=None) -> None:
        """Initialize a shard director client object."""
        self.shard_name = shard_name
        options = [('grpc.max_message_length', 100 * 1024 * 1024)]
        if not tls:
            channel = grpc.insecure_channel(director_uri, options=options)
        else:
            if not (root_ca and key and cert):
                raise Exception('No certificates provided')
            try:
                with open(root_ca, 'rb') as f:
                    root_ca_b = f.read()
                with open(key, 'rb') as f:
                    key_b = f.read()
                with open(cert, 'rb') as f:
                    cert_b = f.read()
            except FileNotFoundError as exc:
                raise Exception(f'Provided certificate file is not exist: {exc.filename}')

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_ca_b,
                private_key=key_b,
                certificate_chain=cert_b
            )
            channel = grpc.secure_channel(director_uri, credentials, options=options)
        self.stub = director_pb2_grpc.FederationDirectorStub(channel)

    def report_shard_info(self, shard_descriptor) -> bool:
        """Report shard info to the director."""
        logger.info('Send report AcknowledgeShard')
        # True considered as successful registration
        shard_info = director_pb2.ShardInfo(
            shard_description=shard_descriptor.dataset_description,
            n_samples=len(shard_descriptor),
            sample_shape=shard_descriptor.sample_shape,
            target_shape=shard_descriptor.target_shape
        )

        shard_info.node_info.CopyFrom(self._get_node_info())

        acknowledgement = self.stub.AcknowledgeShard(shard_info)
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

    def _get_experiment_data(self):
        """Generate the experiment data request."""
        yield director_pb2.WaitExperimentRequest(collaborator_name=self.shard_name)

    def _get_node_info(self):
        """Generate a node info message."""
        return director_pb2.NodeInfo(name=self.shard_name)

    def send_health_check(self, collaborator_name, is_experiment_running, valid_duration):
        """Send envoy health check."""
        status = director_pb2.CollaboratorStatus(
            name=collaborator_name,
            is_experiment_running=is_experiment_running,
        )
        status.valid_duration.seconds = valid_duration
        logger.debug(f'Sending health check status: {status}')

        return self.stub.CollaboratorHealthCheck(status)


class DirectorClient:
    """Director client class for users."""

    def __init__(self, client_id, director_uri, tls=True,
                 root_ca=None, key=None, cert=None) -> None:
        """Initialize director client object."""
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        if not tls:
            channel = grpc.insecure_channel(director_uri, options=channel_opt)
        else:
            if not (root_ca and key and cert):
                raise Exception('No certificates provided')
            try:
                with open(root_ca, 'rb') as f:
                    root_ca_b = f.read()
                with open(key, 'rb') as f:
                    key_b = f.read()
                with open(cert, 'rb') as f:
                    cert_b = f.read()
            except FileNotFoundError as exc:
                raise Exception(f'Provided certificate file is not exist: {exc.filename}')

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_ca_b,
                private_key=key_b,
                certificate_chain=cert_b
            )

            channel = grpc.secure_channel(director_uri, credentials, options=channel_opt)
        self.stub = director_pb2_grpc.FederationDirectorStub(channel)

        self.client_id = client_id
        self.header = director_pb2.RequestHeader(sender=self.client_id)

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
                    header=self.header,
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
        resp = self.stub.GetDatasetInfo(director_pb2.GetDatasetInfoRequest(header=self.header))
        return resp.sample_shape, resp.target_shape

    def _get_trained_model(self, experiment_name, model_type):
        """Get trained model RPC."""
        get_model_request = director_pb2.GetTrainedModelRequest(
            header=self.header,
            experiment_name=experiment_name,
            model_type=model_type)
        model_proto_response = self.stub.GetTrainedModel(get_model_request)
        tensor_dict, _ = deconstruct_model_proto(
            model_proto_response.model_proto,
            NoCompressionPipeline()
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
        request = director_pb2.StreamMetricsRequest(
            header=self.header,
            experiment_name=experiment_name)
        for metric_message in self.stub.StreamMetrics(request):
            yield {
                'metric_origin': metric_message.metric_origin,
                'task_name': metric_message.task_name,
                'metric_name': metric_message.metric_name,
                'metric_value': metric_message.metric_value,
                'round': metric_message.round
            }

    def remove_experiment_data(self, name):
        """Remove experiment data RPC."""
        request = director_pb2.RemoveExperimentRequest(
            header=self.header,
            experiment_name=name)
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
                'valid_duration': envoy.valid_duration
            }
        return result
