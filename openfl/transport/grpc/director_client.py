# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director clients module."""

import logging
import os
import shutil
import time
from subprocess import check_call
from sys import executable

import grpc

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto


logger = logging.getLogger(__name__)


class ShardDirectorClient:
    """The internal director client class."""

    def __init__(self, director_uri, shard_name, disable_tls=False,
                 root_ca=None, key=None, cert=None) -> None:
        """Initialize a shard director client object."""
        self.shard_name = shard_name
        options = [('grpc.max_message_length', 100 * 1024 * 1024)]
        if disable_tls:
            channel = grpc.insecure_channel(director_uri, options=options)
        else:
            if not (root_ca and key and cert):
                raise Exception('No certificates provided')
            with open(root_ca, 'rb') as f:
                root_ca_b = f.read()
            with open(key, 'rb') as f:
                key_b = f.read()
            with open(cert, 'rb') as f:
                cert_b = f.read()

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

    def get_experiment_data(self):
        """Get an experiment data from the director."""
        logger.info('Send WaitExperiment request')
        response_iter = self.stub.WaitExperiment(self._get_experiment_data())
        logger.info('WaitExperiment response has received')
        experiment_name = None
        for response in response_iter:
            experiment_name = response.experiment_name
        if not experiment_name:
            raise Exception('No experiment')
        logger.info(f'Request experiment {experiment_name}')
        request = director_pb2.GetExperimentDataRequest(
            experiment_name=experiment_name,
            collaborator_name=self.shard_name
        )
        response_iter = self.stub.GetExperimentData(request)

        self.create_workspace(experiment_name, response_iter)

        return experiment_name

    def remove_workspace(self, experiment_name):
        """Remove the workspace."""
        shutil.rmtree(experiment_name)

    @staticmethod
    def create_workspace(experiment_name, response_iter):
        """Create a collaborator workspace for the experiment."""
        if os.path.exists(experiment_name):
            shutil.rmtree(experiment_name)
        os.makedirs(experiment_name)

        arch_name = f'{experiment_name}/{experiment_name}' + '.zip'
        logger.info(f'arch_name: {arch_name}')
        with open(arch_name, 'wb') as content_file:
            for response in response_iter:
                logger.info(f'Size: {response.size}')
                if response.size == len(response.npbytes):
                    content_file.write(response.npbytes)
                else:
                    raise Exception('Broken archive')

        shutil.unpack_archive(arch_name, experiment_name)
        os.remove(arch_name)

        requirements_filename = f'./{experiment_name}/requirements.txt'

        if os.path.isfile(requirements_filename):
            attempts = 3
            for _ in range(attempts):
                try:
                    check_call([
                        executable, '-m', 'pip', 'install', '-r', requirements_filename],
                        shell=False)
                except Exception as exc:
                    logger.error(f'Failed to install requirements: {exc}')
                    time.sleep(3)
        else:
            logger.error('No ' + requirements_filename + ' file found.')

    def _get_experiment_data(self):
        """Generate the experiment data request."""
        yield director_pb2.WaitExperimentRequest(collaborator_name=self.shard_name)

    def _get_node_info(self):
        """Generate a node info message."""
        return director_pb2.NodeInfo(name=self.shard_name)


class DirectorClient:
    """Director client class for users."""

    def __init__(self, client_id, director_uri, disable_tls, root_ca, key, cert) -> None:
        """Initialize director client object."""
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        if disable_tls:
            channel = grpc.insecure_channel(director_uri, options=channel_opt)
        else:
            with open(root_ca, 'rb') as f:
                root_ca_b = f.read()
            with open(key, 'rb') as f:
                key_b = f.read()
            with open(cert, 'rb') as f:
                cert_b = f.read()

            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_ca_b,
                private_key=key_b,
                certificate_chain=cert_b
            )

            channel = grpc.secure_channel(director_uri, credentials, options=channel_opt)
        self.stub = director_pb2_grpc.FederationDirectorStub(channel)

        self.client_id = client_id
        self.header = director_pb2.RequestHeader(sender=self.client_id)

    def report_shard_info(self, shard_descriptor) -> bool:
        """Report shard info to the director."""
        logger.info('Send report AcknowledgeShard')

    def set_new_experiment(self, name, col_names, arch_path,
                           initial_tensor_dict=None):
        """Send the new experiment to director to launch."""
        logger.info('SetNewExperiment')
        model_proto = None
        if initial_tensor_dict:
            model_proto = construct_model_proto(initial_tensor_dict, 0, NoCompressionPipeline())

        with open(arch_path, 'rb') as arch:
            def st():
                max_buffer_size = (2 * 1024 * 1024)
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

            resp = self.stub.SetNewExperiment(st())
            return resp

    def get_shard_info(self):
        """Request the shard info to the director."""
        resp = self.stub.GetShardsInfo(director_pb2.GetShardsInfoRequest(header=self.header))
        return resp.sample_shape, resp.target_shape

    def request_shard_registry(self):
        """Request a shard registry."""
        resp = self.stub.GetRegisterdShards(director_pb2.GetRegisterdShardsRequest(
            header=self.header))
        return resp.shard_info

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
            yield metric_message

    def remove_experiment_data(self, experiment_name):
        """Remove experiment data RPC."""
        request = director_pb2.RemoveExperimnetRequest(
            header=self.header,
            experiment_name=experiment_name)
        response = self.stub.RemoveExperimentData(request)
        return response.acknowledgement
