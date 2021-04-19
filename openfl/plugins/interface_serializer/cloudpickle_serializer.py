import cloudpickle
from .serializer_interface import Serializer

class Cloudpickle_Serializer(Serializer):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def serialize(object_, filename):
        with open(filename, 'wb') as f:
            cloudpickle.dump(object_, f)
        
    @staticmethod
    def restore_object(filename):
        with open(filename, 'rb') as f:
            return cloudpickle.load(f)
