from openfl.plugins.frameworks_adapters.framework_adapter_interface \
    import FrameworkAdapterPluginInterface


class CustomFrameworkAdapter(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""
        pass

    @staticmethod
    def get_tensor_dict(model, optimizer=None):
        return {'w': model.weights}

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        model.weights = tensor_dict['w']
