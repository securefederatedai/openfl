import dill
from .serializer_interface import Serializer

class Dill_Serializer(Serializer):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def serialize(object, filename):
        with open(filename, 'wb') as f:
            dill.dump(object, f, recurse=True)
        
    @staticmethod
    def restore_object(filename):
        with open(filename, 'rb') as f:
            return dill.load(f)