import importlib

def get_local_grpc_server(framework: str = 'Flower') -> object:
    if framework == 'Flower':
        try:
            module = importlib.import_module('src.grpc.connector.flower.local_grpc_server')
            return module.LocalGRPCServer
        except ImportError:
            print("Flower is not installed.")
            return None