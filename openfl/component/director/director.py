from concurrent.futures import ProcessPoolExecutor
import logging
from grpc import aio
import asyncio
from pathlib import Path
import os
import shutil
from collections import defaultdict

from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.federated import Plan
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols.utils import deconstruct_model_proto

logger = logging.getLogger(__name__)


class Director(director_pb2_grpc.FederationDirectorServicer):

    def __init__(self, sample_shape: list, target_shape: list) -> None:
        # TODO: add working directory
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self.shard_registry = []
        self.experiments = set()
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self.experiment_data = {}
        self.experiments_queue = asyncio.Queue()
        self.executor = ProcessPoolExecutor(max_workers=2)
        self.aggregator_task = None  # TODO: add check if exists and wait on terminate

    async def AcknowledgeShard(self, shard_info, context):
        reply = director_pb2.ShardAcknowledgement(accepted=False)
        # If dataset do not match the data interface of the problem
        if (self.sample_shape != shard_info.sample_shape) or \
                (self.target_shape != shard_info.target_shape):
            return reply

        self.shard_registry.append(shard_info)
        # logger.info('\n\n\nRegistry now looks like this\n\n', self.shard_registry)
        reply.accepted = True
        return reply

    async def SetNewExperiment(self, stream, context):
        logger.info(f'SetNewExperiment request has got {stream}')
        # TODO: add streaming reader
        npbytes = b""
        async for request in stream:
            if request.experiment_data.size == len(request.experiment_data.npbytes):
                npbytes += request.experiment_data.npbytes
            else:
                raise Exception('Bad request')

        logger.info(f'New experiment {request.name} for collaborators {request.collaborator_names}')
        # TODO: save to file
        self.experiment_data[request.name] = npbytes
        tensor_dict = None
        if request.model_proto:
            tensor_dict, _ = deconstruct_model_proto(request.model_proto, NoCompressionPipeline())

        # TODO: add a logic with many experiments
        for col_name in request.collaborator_names:
            queue = self.col_exp_queues[col_name]
            await queue.put(request.name)
        self.create_workspace(request.name, npbytes)
        self._run_aggregator(tensor_dict, request.name)
        # loop = asyncio.get_event_loop()
        # fut = loop.run_in_executor(self.executor, self._run_aggregator, tensor_dict, request.name)
        # await fut
        # future = self.executor.submit(self._run_aggregator, tensor_dict, request.name)
        # self.aggregator_task = future

        return director_pb2.Response(accepted=True)

    async def GetExperimentData(self, request, context):
        # experiment_data = preparations_pb2.ExperimentData()
        # with open(experiment_name + '.zip', 'rb') as content_file:
        #     content = content_file.read()
        #     # TODO: add size filling
        #     # TODO: add experiment name field
        #     # TODO: rename npbytes to data
        content = self.experiment_data.get(request.experiment_name, b'')
        logger.info(f'Content length: {len(content)}')
        max_buffer_size = (2 * 1024 * 1024)

        for i in range(0, len(content), max_buffer_size):
            chunk = content[i:i + max_buffer_size]
            logger.info(f'Send {len(chunk)} bytes')
            yield director_pb2.ExperimentData(size=len(chunk), npbytes=chunk)

    # Await prob
    async def WaitExperiment(self, request_iterator, context):
        logger.info('Request WaitExperiment has got!')
        async for msg in request_iterator:
            logger.info(msg)
        queue = self.col_exp_queues[msg.collaborator_name]
        experiment_name = await queue.get()
        logger.info(f'Experiment {experiment_name} was prepared')

        yield director_pb2.WaitExperimentResponse(experiment_name=experiment_name)

    async def GetShardsInfo(self, request, context):
        logger.info('Request GetShardsInfo has got!')
        resp = director_pb2.ShardInfo(
            sample_shape=self.sample_shape,
            target_shape=self.target_shape
        )
        return resp

    async def GetRegisterdShards(self, request, context):
        logger.info('Request GetRegisterdShards has got!')
        resp = director_pb2.GetRegisterdShardsResponse(
            shard_info=self.shard_registry
        )
        return resp

    @staticmethod
    def create_workspace(experiment_name, npbytes):
        if os.path.exists(experiment_name):
            shutil.rmtree(experiment_name)
        os.makedirs(experiment_name)

        arch_name = f'{experiment_name}/{experiment_name}' + '.zip'
        logger.info(f'arch_name: {arch_name}')
        with open(arch_name, 'wb') as content_file:
            content_file.write(npbytes)

        shutil.unpack_archive(arch_name, experiment_name)

    def _run_aggregator(
            self,
            initial_tensor_dict,
            experiment_name,
            plan='plan/plan.yaml',
    ):  # TODO: path params, change naming
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')
        plan = Plan.Parse(plan_config_path=Path(plan))
        plan.authorized_cols = list(self.col_exp_queues.keys())

        logger.info('🧿 Starting the Aggregator Service.')
        server = plan.interactive_api_get_server(
            initial_tensor_dict,
            chain='',
            certificate='',
            private_key='')

        logger.error(server.serve())
        server.wait_for_termination()


async def serve(*args, **kwargs):
    logging.basicConfig()
    server = aio.server()
    director_pb2_grpc.add_FederationDirectorServicer_to_server(
        Director(*args, **kwargs), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logger.info(f'Starting server on {listen_addr}')
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
