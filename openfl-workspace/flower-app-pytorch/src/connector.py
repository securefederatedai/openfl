import signal
import sys
from logging import getLogger
from abc import ABC, abstractmethod

class Connector(ABC):
    """
    Abstract base class for managing a server process of an external federated learning framework
    and the connection with OpenFL's server.
    """

    def __init__(self, component_name: str = "Base", **kwargs):
        """
        Initialize the BaseConnector.
        
        Args:
            command (list[str]): The command to run the server process.
            component_name (str): The name of the specific Connector component being used.
        """
        self.logger = getLogger(__name__)
        self.component_name = component_name
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
    
    def print_connector_info(self):
        """Print information indicating which Connector component is being used."""
        self.logger.info(f"OpenFL Connector Enabled: {self.component_name}")

    def _handle_sigint(self, signum, frame):
        """Handle the SIGINT signal (Ctrl+C) to cleanly stop the server process and its children."""
        self.logger.info("[OpenFL Connector] SIGINT received. Terminating server process...")
        self.stop()
        sys.exit(0)