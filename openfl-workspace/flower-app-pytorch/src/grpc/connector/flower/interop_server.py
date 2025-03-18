import logging
import threading
import queue
import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from src.grpc.connector.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message
from multiprocessing import cpu_count
import signal
import psutil
import time

logger = logging.getLogger(__name__)


class FlowerInteropServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    """
    FlowerInteropServer is a gRPC server that handles requests from the Flower SuperNode
    and forwards them to the OpenFL Client. It uses a queue-based system to
    ensure that requests are processed sequentially, preventing concurrent
    request handling issues.
    """

    def __init__(self, openfl_client, collaborator_name):
        """
        Initialize.

        Args:
            openfl_client: An instance of the OpenFL Client.
            collaborator_name: The name of the collaborator.
        """
        self.openfl_client = openfl_client
        self.collaborator_name = collaborator_name
        self.end_experiment_callback = None
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.server = None
        self.termination_event = threading.Event()
        self.port = None

    def set_end_experiment_callback(self, callback):
        self.end_experiment_callback = callback

    def start_server(self, local_server_port):
        """Starts the gRPC server."""
        self.server = grpc.server(ThreadPoolExecutor(max_workers=cpu_count()))
        grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(self, self.server)
        self.port = self.server.add_insecure_port(f'[::]:{local_server_port}')
        self.server.start()
        logger.info(f"OpenFL local gRPC server started, listening on port {self.port}.")

    def get_port(self):
        # Return the port that was assigned
        return self.port

    def stop_server(self):
        """Stops the gRPC server."""
        if self.server:
            logger.info("Shutting down local gRPC server...")
            self.server.stop(0)
            logger.info("local gRPC server stopped.")
            self.termination_event.set()

    def SendReceive(self, request, context):
        """ Handles incoming gRPC requests by putting them into the request queue and waiting for the response.
        Args:
            request: The incoming gRPC request.
            context: The gRPC context.
        Returns:
            The response from the OpenFL server.
        """
        response_queue = queue.Queue()
        self.request_queue.put((request, response_queue))
        return response_queue.get()

    def process_queue(self):
        """
        Continuously processes requests from the request queue. Each request is
        sent to the OpenFL server, and the response is put into the corresponding
        response queue.
        """
        while True:
            request, response_queue = self.request_queue.get()
            openfl_request = flower_to_openfl_message(request)

            # Send request to the OpenFL server
            openfl_response = self.openfl_client.send_message_to_server(openfl_request, self.collaborator_name)

            # Check to end experiment
            if hasattr(openfl_response, 'metadata'):
                if openfl_response.metadata['end_experiment'] == 'True':
                    self.end_experiment_callback()

            # Send response to Flower client
            flower_response = openfl_to_flower_message(openfl_response)
            response_queue.put(flower_response)
            self.request_queue.task_done()

    def handle_signals(self, supernode_process):
        """Sets up signal handlers for graceful shutdown."""
        def signal_handler(_sig, _frame):
            self.terminate_supernode_process(supernode_process)
            self.stop_server()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def terminate_supernode_process(self, supernode_process):
        """Terminates the SuperNode process."""
        if supernode_process.poll() is None:
            try:
                main_subprocess = psutil.Process(supernode_process.pid)
                client_app_processes = main_subprocess.children(recursive=True)

                for client_app_process in client_app_processes:
                    self.terminate_process(client_app_process)

                self.terminate_process(main_subprocess)
                logger.info("SuperNode process terminated.")

            except Exception as e:
                logger.debug(f"Error during graceful shutdown: {e}")
                time.sleep(10)
                supernode_process.kill()
                logger.info("SuperNode process terminated.")
        else:
            logger.info("SuperNode process already terminated.")

    def terminate_process(self, process, timeout=5):
        """Helper function to terminate a process gracefully."""
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            logger.debug(f"Timeout expired while waiting for process {process.pid} to terminate. Killing the process.")
            process.kill()
        except psutil.NoSuchProcess:
            logger.debug(f"Process {process.pid} does not exist. Skipping.")
            pass
