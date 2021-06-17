import logging
import os
import shutil

import grpc

from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols.utils import construct_model_proto

logger = logging.getLogger(__name__)


class ShardDirectorClient:
    def __init__(self, director_uri, shard_name) -> None:
        self.shard_name = shard_name
        options = [('grpc.max_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel(director_uri, options=options)

        self.stub = director_pb2_grpc.FederationDirectorStub(channel)

    def report_shard_info(self, shard_descriptor) -> bool:
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
        logger.info('Send WaitExperiment request')
        response_iter = self.stub.WaitExperiment(self._get_experiment_data())
        logger.info(f'WaitExperiment response has received')
        # TODO: seperate into two resuests (get status and get file)
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

    @staticmethod
    def create_workspace(experiment_name, response_iter):
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

    def _get_experiment_data(self):
        yield director_pb2.WaitExperimentRequest(collaborator_name=self.shard_name)

    def _get_node_info(self):
        return director_pb2.NodeInfo(name=self.shard_name)


class DirectorClient:
    def __init__(self, director_uri) -> None:
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel(director_uri, options=channel_opt)
        self.stub = director_pb2_grpc.FederationDirectorStub(channel)

    def set_new_experiment(self, name, col_names, arch_path,
                           model_interface=None, fl_experiment=None):
        logger.info('SetNewExperiment')
        model_proto = None
        if model_interface:
            initial_tensor_dict = fl_experiment._get_initial_tensor_dict(model_interface)
            model_proto = construct_model_proto(initial_tensor_dict, 0, NoCompressionPipeline())

        with open(arch_path, 'rb') as arch:
            def st():
                max_buffer_size = (2 * 1024 * 1024)
                chunk = arch.read(max_buffer_size)
                while chunk != b"":
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

            resp = self.stub.SetNewExperiment(st())
            return resp

    def get_shard_info(self):
        resp = self.stub.GetShardsInfo(director_pb2.GetShardsInfoRequest())
        return resp.sample_shape, resp.target_shape

    def request_shard_registry(self):
        resp = self.stub.GetRegisterdShards(director_pb2.GetRegisterdShardsRequest())
        return resp.shard_info
