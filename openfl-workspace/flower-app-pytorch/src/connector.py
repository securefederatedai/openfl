import signal
import sys
from logging import getLogger
from abc import ABC, abstractmethod

class Connector(ABC):
    """
    Abstract base class for managing a server process of an external federated learning framework
    and the connection with OpenFL's server.
    """

    def __init__(self, **kwargs):
        """
        Initialize the BaseConnector.

        Args:
            command (list[str]): The command to run the server process.
        """
        self.logger = getLogger(__name__)
        self.local_grpc_client = None

        # Register signal handler for clean termination
        signal.signal(signal.SIGINT, self._handle_sigint)

    @abstractmethod
    def start(self):
        """Start the server process with the provided command."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the server process if it is running."""
        pass

    def get_local_grpc_client(self):
        """Get the local gRPC client."""
        return self.local_grpc_client

    def _handle_sigint(self, signum, frame):
        """Handle the SIGINT signal (Ctrl+C) to cleanly stop the server process and its children."""
        self.logger.info("[OpenFL Connector] SIGINT received. Terminating server process...")
        self.stop()
        sys.exit(0)
