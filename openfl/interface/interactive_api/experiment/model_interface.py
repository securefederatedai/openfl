class ModelInterface:
    """
    Registers model graph and optimizer.

    To be serialized and sent to collaborator nodes

    This is the place to determine correct framework adapter
        as they are needed to fill the model graph with trained tensors.

    There is no support for several models / optimizers yet.
    """

    def __init__(self, model, optimizer, framework_plugin) -> None:
        """
        Initialize model keeper.

        Tensors in provided graphs will be used for
        initialization of the global model.

        Arguments:
        model: Union[tuple, graph]
        optimizer: Union[tuple, optimizer]
        """
        self.model = model
        self.optimizer = optimizer
        self.framework_plugin = framework_plugin

    def provide_model(self):
        """Retrieve model."""
        return self.model

    def provide_optimizer(self):
        """Retrieve optimizer."""
        return self.optimizer
