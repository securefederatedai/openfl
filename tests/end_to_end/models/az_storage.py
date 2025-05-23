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

    def __init__(
        self,
        host=defaults.AZURE_STORAGE_HOST,
        port=defaults.AZURE_STORAGE_PORT,
        account_name=defaults.AZURE_STORAGE_ACCOUNT_NAME,
        account_key=defaults.AZURE_STORAGE_ACCOUNT_KEY,
        endpoints_protocol=defaults.AZURE_STORAGE_ENDPOINTS_PROTOCOL,
    ):
        """
        Initialize the AzureStorage class
        Args:
            account_name (str): Azure Storage account name
            account_key (str): Azure Storage account key
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

    def start_azurite_container(self):
        """
        Start the Azurite container for local testing.
        """
        try:
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

    def upload_data_to_container(self, container_name, data_path: Path):
        """
        Upload a file to Azure Storage
        Args:
            container_name (str): Name of the container
            blob_name (str): Name of the blob
            file_path (str): Path to the file
        """
        try:
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
