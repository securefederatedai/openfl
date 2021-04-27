class FrameworkAdapterPlugin:
    def __init__(self) -> None:
        pass

    @staticmethod
    def serialization_setup():
        pass

    @staticmethod
    def get_tensor_dict(model, optimizer=None):
        raise NotImplementedError

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        raise NotImplementedError
