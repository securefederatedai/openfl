from openfl.callbacks.callback import Callback


class TensorBoard(Callback):
    """Callback to log summaries for visualization on TensorBoard.

    Note: A tensorflow installation is required to use this callback.
    """

    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
