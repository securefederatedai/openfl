# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from azure.storage.blob import BlobServiceClient
from pathlib import Path

import tests.end_to_end.utils.defaults as defaults
import tests.end_to_end.utils.docker_helper as docker_helper
import tests.end_to_end.utils.exceptions as ex

# Suppress Azure SDK and urllib3 info/debug logs
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies._universal").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


class AzureStorage():
    """
    Class to handle Azure Storage
    """

    def __init__(self, host, port, account_name, account_key, endpoints_protocol):
        """
        Initialize the AzureStorage class
        Args:
            host (str): Azure Storage host
            port (int): Azure Storage port
            account_name (str): Azure Storage account name
            account_key (str): Azure Storage account key
            endpoints_protocol (str): Protocol for the endpoints (http or https)
        """
        self.host = host
        self.port = port
        self.account_name = account_name
        self.account_key = account_key
        self.endpoints_protocol = endpoints_protocol
        self.blob_endpoint = f"{self.endpoints_protocol}://{self.host}:{self.port}/{self.account_name}"
        self.blob_service_client = BlobServiceClient(
            account_url=self.blob_endpoint,
            credential=self.account_key,
        )
        self.connection_string = f"DefaultEndpointsProtocol={endpoints_protocol};AccountName={account_name};AccountKey={account_key};BlobEndpoint={self.blob_endpoint};"

    def create_container(self, container_name):
        """
        Create a container in Azure Storage
        Args:
            container_name (str): Name of the container
        """
        try:
            container_client = self.blob_service_client.create_container(container_name)
            log.info(f"Container {container_name} created successfully")
        except Exception as e:
            log.error(f"Failed to create container: {e}")
            raise e
        return container_client

    def delete_container(self, container_name):
        """
        Delete a container in Azure Storage
        Args:
            container_name (str): Name of the container
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            container_client.delete_container()
            log.info(f"Container {container_name} deleted successfully")
        except Exception as e:
            log.error(f"Failed to delete container: {e}")
            raise e
        return True

    def upload_data_to_container(self, container_name, data_path: Path):
        """
        Upload a file to Azure Storage.
        Assumption - data_path contains the file to be uploaded.
        Args:
            container_name (str): Name of the container
            file_path (str): Path to the file
        """
        try:
            # Verify data path
            if not data_path.exists() or not data_path.is_dir():
                raise ValueError(f"Expected {data_path} to be a directory, but it does not exist or is not a directory.")

            if not any(data_path.iterdir()):
                raise ValueError(f"Directory {data_path} is empty. Nothing to upload.")

            container_client = self.blob_service_client.get_container_client(container_name)
            num = 0
            for file_path in data_path.rglob("*"):
                if file_path.is_file():
                    blob_name = str(file_path.relative_to(data_path)).replace("\\", "/")
                    with open(file_path, "rb") as data:
                        container_client.upload_blob(blob_name, data, overwrite=True)
                    num += 1
            log.info(f"Uploaded {num} files to {container_name}: {blob_name}")
        except Exception as e:
            log.error(f"Failed to upload file: {e}")
            raise e
        return True


class AzuriteStorage(AzureStorage):
    """
    Azurite is an emulator for local Azure Storage development.
    This class provides methods to start, stop, and manage the Azurite container.
    """
    def __init__(
        self,
        host=defaults.AZURE_STORAGE_HOST,
        port=defaults.AZURE_STORAGE_PORT,
        account_name=defaults.AZURE_STORAGE_ACCOUNT_NAME,
        account_key=defaults.AZURE_STORAGE_ACCOUNT_KEY,
        endpoints_protocol=defaults.AZURE_STORAGE_ENDPOINTS_PROTOCOL,
    ):
        """
        Initialize the AzuriteStorage class
        Args:
            account_name (str): Azure Storage account name
            account_key (str): Azure Storage account key
        """
        super().__init__(host, port, account_name, account_key, endpoints_protocol)

    def start_azurite_container(self):
        """
        Start the Azurite container for local testing.
        """
        try:
            # Stop and remove if already running or remove if exited
            is_container_present = self.is_azurite_container_present()
            if is_container_present:
                log.info("Azurite container is present in either running or exited state. Stopping/removing for a fresh start.")
                self.stop_azurite_container()
            client = docker_helper.get_docker_client()
            container = client.containers.run(
                "mcr.microsoft.com/azure-storage/azurite",
                detach=True,
                ports={"10000/tcp": 10000, "10001/tcp": 10001, "10002/tcp": 10002},
                name="azurite",
            )
            log.info(f"Azurite container started with ID: {container.id}")
        except Exception as e:
            raise ex.DockerException(f"Error starting Azurite container: {e}")
        return container

    def stop_azurite_container(self):
        """
        Stop the Azurite container.
        """
        try:
            client = docker_helper.get_docker_client()
            container = client.containers.get("azurite")
            container.stop()
            container.remove()
            log.info("Azurite container stopped and removed successfully")
        except Exception as e:
            raise ex.DockerException(f"Error stopping Azurite container: {e}")
        return True
    
    def is_azurite_container_present(self):
        """Check if Azurite container is present."""
        try:
            log.info("Checking if Azurite container is present...")
            client = docker_helper.get_docker_client()
            container = client.containers.get("azurite")
            if container.status in ("running", "exited"):
                return container
        except Exception:
            pass
        return None
