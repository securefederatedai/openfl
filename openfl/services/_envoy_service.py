from concurrent import futures

import logging
import grpc

from openfl.protocols import preparations_pb2
from openfl.protocols import preparations_pb2_grpc


class Envoy(preparations_pb2_grpc.EnvoyServicer):
    def __init__(self, sample_shape, target_shape) -> None:
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self.shard_registry = list()

    def SetWorkspace(self, workspace_info, context):
        reply = preparations_pb2.Response(accepted=False)

        self.extract_workspace(workspace_info)
        reply.accepted = True
        return reply

    def RunExperiment(self, experience_info, context):
        reply = preparations_pb2.Response(accepted=False)

        # run experiment

        reply.accepted = True
        return reply

    def extract_workspace(self, workspace_info):
        pass

    def serve(self):
        logging.basicConfig()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        preparations_pb2_grpc.add_EnvoyServicer_to_server(self, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()
