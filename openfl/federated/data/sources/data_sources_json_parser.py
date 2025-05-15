# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import os
from pathlib import Path

from openfl.federated.data.sources.azure_blob_data_source import AzureBlobDataSource
from openfl.federated.data.sources.local_data_source import LocalDataSource
from openfl.federated.data.sources.s3_data_source import S3DataSource
from openfl.federated.data.sources.verifiable_dataset_info import VerifiableDatasetInfo
from openfl.utilities.path_check import is_directory_traversal


class DataSourcesJsonParser:
    @staticmethod
    def parse(json_string: str) -> VerifiableDatasetInfo:
        """
        Parse a JSON string into a dictionary.

        Args:
            json_string (str): The JSON string to parse.

        Returns:
            VerifiableDatasetInfo: An instance of VerifiableDatasetInfo containing
            the parsed data sources.
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        datasources = DataSourcesJsonParser.process_data_sources(data)
        if not datasources:
            raise ValueError("No data sources were found.")
        return VerifiableDatasetInfo(
            data_sources=datasources,
            label="",
        )

    @staticmethod
    def process_data_sources(data):
        """Process and validate data sources."""
        cwd = os.getcwd()
        datasources = []
        for source_name, source_info in data.items():
            source_type = source_info.get("type", None)
            if source_type is None:
                raise ValueError(f"Missing 'type' key in data source configuration: {source_info}")
            params = source_info.get("params", {})
            if source_type == "local":
                datasources.append(
                    DataSourcesJsonParser.process_local_source(source_name, params, cwd)
                )
            elif source_type == "s3":
                datasources.append(DataSourcesJsonParser.process_s3_source(source_name, params))
            elif source_type == "azure_blob":
                datasources.append(
                    DataSourcesJsonParser.process_azure_blob_source(source_name, params)
                )
        return [ds for ds in datasources if ds]

    @staticmethod
    def process_local_source(source_name, params, cwd):
        """Process a local data source."""
        path = params.get("path", None)
        if not path:
            raise ValueError(f"Missing 'path' parameter for local data source '{source_name}'")
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, cwd)
        if rel_path and not is_directory_traversal(rel_path):
            return LocalDataSource(source_name, rel_path, base_path=Path("."))
        else:
            raise ValueError(f"Invalid path for local data source '{source_name}': {path}.")

    @staticmethod
    def process_s3_source(source_name, params):
        """Process an S3 data source."""
        required_fields = ["uri", "access_key_env_name", "secret_key_env_name", "secret_name"]
        missing_fields = set(required_fields) - set(params.keys())
        if missing_fields:
            raise Exception(
                f"Missing required fields: {', '.join(missing_fields)} "
                f"for S3 data source '{source_name}'"
            )
        return S3DataSource(
            name=source_name,
            uri=params["uri"],
            endpoint=params.get("endpoint") or None,
            access_key_env_name=params["access_key_env_name"],
            secret_key_env_name=params["secret_key_env_name"],
            secret_name=params["secret_name"],
        )

    @staticmethod
    def process_azure_blob_source(source_name, params):
        """Process an Azure Blob data source."""
        required_fields = ["connection_string", "container_name"]
        missing_fields = set(required_fields) - set(params.keys())
        if missing_fields:
            raise Exception(
                f"Missing required fields: {', '.join(missing_fields)} "
                f"for Azure Blob data source '{source_name}'"
            )
        return AzureBlobDataSource(
            name=source_name,
            connection_string=params["connection_string"],
            container_name=params["container_name"],
            folder_prefix=params.get("folder_prefix") or "",
        )
