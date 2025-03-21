import importlib

def get_interop_server(framework: str = 'Flower') -> object:
    if framework == 'Flower':
        try:
            module = importlib.import_module('src.grpc.connector.flower.interop_server')
            return module.FlowerInteropServer
        except ImportError:
            print("Flower is not installed.")
            return None
