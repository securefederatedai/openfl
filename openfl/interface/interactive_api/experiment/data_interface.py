class DataInterface:
    """
    The class to define dataloaders.

    In the future users will have to adapt `unified data interface hook`
        in their dataloaders.
    For now, we can provide `data_path` variable on every collaborator node
        at initialization time for dataloader customization
    """

    def __init__(self, **kwargs):
        """Initialize DataLoader."""
        self.kwargs = kwargs

    @property
    def shard_descriptor(self):
        """Return shard descriptor."""
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        raise NotImplementedError

    def get_train_loader(self, **kwargs):
        """Output of this method will be provided to tasks with optimizer in contract."""
        raise NotImplementedError

    def get_valid_loader(self, **kwargs):
        """Output of this method will be provided to tasks without optimizer in contract."""
        raise NotImplementedError

    def get_train_data_size(self):
        """Information for aggregation."""
        raise NotImplementedError

    def get_valid_data_size(self):
        """Information for aggregation."""
        raise NotImplementedError
