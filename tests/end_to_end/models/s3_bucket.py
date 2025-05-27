# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import time
import signal
import socket
import shutil
import atexit
import boto3
import logging
from botocore.client import Config
from botocore.exceptions import ClientError
import fnmatch
from pathlib import Path

import tests.end_to_end.utils.defaults as defaults

log = logging.getLogger(__name__)


class MinioServer():
    """
    A class to manage MinIO server operations.
    This class provides methods to start, stop, and check the status of a MinIO server.
    """
    def __init__(
        self,
        access_key=defaults.MINIO_ROOT_USER,
        secret_key=defaults.MINIO_ROOT_PASSWORD,
        minio_url=defaults.MINIO_URL,
        minio_console_url=defaults.MINIO_CONSOLE_URL,
    ):
        """
        Initialize MinIO server with connection details.

        Args:
            access_key: MinIO access key (default: from instance)
            secret_key: MinIO secret key (default: from instance)
            minio_url: MinIO server URL (default: from instance)
            minio_console_url: MinIO console URL (default: from instance)
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.minio_url = minio_url.split("://")[-1]
        self.minio_console_url = minio_console_url.split("://")[-1]

    def is_minio_server_running(self, port=9000):
        """
        Check if a MinIO server is running on the specified host and port.

        Args:
            port: Port number (default: 9000)

        Returns:
            bool: True if MinIO server is running, False otherwise
        """
        try:
            check_cmd = ['lsof', '-i', f':{port}', '-t']
            output = subprocess.check_output(check_cmd, universal_newlines=True).strip()
            if output:
                pids = [int(pid) for pid in output.split()]
                log.info(f"Port {port} is in use (lsof check), PID(s): {pids}")
                return pids
        except Exception:
            pass
        return None

    def start_minio_server(self, data_dir):
        """
        Start a MinIO server as a subprocess.

        Args:
            data_dir: Directory to store data

        Returns:
            subprocess.Popen: The process object for the MinIO server
        """
        # Use instance values if not provided

        # Parse address to get host and port
        try:
            host, port = self.minio_url.split(':')
            port = int(port)
        except ValueError:
            host = 'localhost'
            port = 9001

        # Check if MinIO server is already running
        running = self.is_minio_server_running(port)
        if running:
            log.info("MinIO server already running. Cleaning up for fresh start.")

            if isinstance(running, list):
                self._kill_processes(running)
            else:
                log.warning("MinIO server running but PID not found. Please check manually.")
            
            # Wait for port to be released
            if not self._wait_for_port_release(port, host):
                log.error("Port is still in use. Cannot start MinIO server.")
                return None

        # Throw error if data_dir is not provided
        if data_dir is None:
            log.error("Data directory is required to start MinIO server.")
            return None

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Check if minio is installed
        minio_path = shutil.which("minio")
        if minio_path is None:
            log.error("MinIO server not found. Please install MinIO first.")
            log.warning("You can download it from: https://min.io/download")
            return None

        # Set environment variables for the current process as well as the subprocess
        # This is important for MinIO to pick up the access and secret keys
        # and for the subprocess to inherit them
        env = os.environ.copy()
        env["MINIO_ROOT_USER"] = os.environ["MINIO_ROOT_USER"] = self.access_key
        env["MINIO_ROOT_PASSWORD"] = os.environ["MINIO_ROOT_PASSWORD"] = self.secret_key

        # Start MinIO server
        cmd = [
            minio_path,
            "server",
            data_dir,
            "--address",
            self.minio_url,
            "--console-address",
            self.minio_console_url,
        ]
        log.info(
            "Starting MinIO server with below configurations:"
            f"\n  - Data Directory: {data_dir}"
            f"\n  - Address: {self.minio_url}"
            f"\n  - Console Address: {self.minio_console_url}"
        )

        # Start the process
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Register a function to stop the server at exit
        def stop_server():
            if process.poll() is None:  # If process is still running
                log.info("Stopping MinIO server...")
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        atexit.register(stop_server)

        # Wait for server to start
        time.sleep(2)

        # Check if server started successfully
        if process.poll() is not None:
            # Process exited already
            out, err = process.communicate()
            log.error("Failed to start MinIO server:")
            log.info(f"STDOUT: {out}")
            log.error(f"STDERR: {err}")
            return None

        log.info("MinIO server started successfully.")
        return process

    def _kill_processes(self, pids):
        """Kill processes by PID (SIGTERM, then SIGKILL if needed)."""
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log.info(f"Killed MinIO process with PID {pid} (SIGTERM)")
                time.sleep(1)
                # Check if process is still alive
                try:
                    os.kill(pid, 0)
                    # Still alive, force kill
                    os.kill(pid, signal.SIGKILL)
                    log.info(f"Force killed MinIO process with PID {pid} (SIGKILL)")
                except OSError:
                    # Process is gone
                    pass
            except Exception as e:
                log.warning(f"Could not kill PID {pid}: {e}")
        time.sleep(2)  # Give time for processes to terminate

    def _wait_for_port_release(self, port, host="127.0.0.1", timeout=10):
        """Wait until the port is free, or timeout (seconds) is reached."""
        waited = 0
        while waited < timeout:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((host, port)) != 0:
                    return True  # Port is free
            log.info(f"Waiting for port {port} to be released...")
            time.sleep(1)
            waited += 1
        log.error(f"Port {port} is still in use after waiting {timeout} seconds.")
        return False


class S3Bucket():
    """
    A class to manage S3 bucket operations using boto3.
    This class provides methods to create, delete, upload, download,
    and list objects in S3 buckets, as well as manage MinIO server.
    """

    def __init__(
        self,
        endpoint_url=defaults.MINIO_URL,
        access_key=defaults.MINIO_ROOT_USER,
        secret_key=defaults.MINIO_ROOT_PASSWORD,
        region=None,
    ):
        """
        Initialize S3Helper with connection details.

        Args:
            endpoint_url: The S3 endpoint URL (default: http://localhost:9000 for MinIO)
            access_key: The access key (if None, uses MINIO_ROOT_USER env variable)
            secret_key: The secret key (if None, uses MINIO_ROOT_PASSWORD env variable)
            region: The region name (default: None, required by boto3 but not used by MinIO or on local server)
        """
        self.endpoint_url = endpoint_url
        self.access_key = access_key or os.environ.get("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = secret_key or os.environ.get(
            "MINIO_ROOT_PASSWORD", "minioadmin"
        )
        self.region = region

        # Extract host and port from endpoint_url
        url_parts = self.endpoint_url.split('://')[-1].split(':')
        self.minio_host = url_parts[0]
        self.minio_port = int(url_parts[1]) if len(url_parts) > 1 else 9000

        # Set default URLs
        self.minio_url = f"{self.minio_host}:{self.minio_port}"
        self.minio_console_url = f"{self.minio_host}:{self.minio_port + 1}"

        # Initialize S3 client
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=self.region,
        )

    def create_bucket(self, bucket_name):
        """
        Create a new bucket if it doesn't exist.

        Args:
            bucket_name: Name of the bucket to create

        Returns:
            bool: True if bucket was created or already exists, False on error
        """
        try:
            # Check if bucket already exists
            self.client.head_bucket(Bucket=bucket_name)
            log.info(f"Bucket {bucket_name} already exists. Deleting all objects in the bucket.")
            self.delete_all_objects(bucket_name)
            return True
        except ClientError as e:
            # If bucket doesn't exist, create it
            if e.response["Error"]["Code"] == "404":
                try:
                    self.client.create_bucket(Bucket=bucket_name)
                    log.info(f"Bucket {bucket_name} created successfully.")
                    return True
                except ClientError as create_error:
                    log.error(f"Error creating bucket: {create_error}")
                    return False
            else:
                log.error(f"Error checking bucket: {e}")
                return False

    def delete_bucket(self, bucket_name, force=False):
        """
        Delete a bucket.

        Args:
            bucket_name: Name of the bucket to delete
            force: If True, delete all objects in the bucket before deletion

        Returns:
            bool: True if bucket was deleted, False on error
        """
        try:
            if force:
                # Delete all objects in the bucket first
                self.delete_all_objects(bucket_name)

            # Delete the bucket
            self.client.delete_bucket(Bucket=bucket_name)
            log.info(f"Bucket {bucket_name} deleted successfully.")
            return True
        except ClientError as e:
            log.error(f"Error deleting bucket {bucket_name}: {e}")
            return False

    def list_buckets(self):
        """
        List all buckets.

        Returns:
            list: List of bucket names
        """
        try:
            response = self.client.list_buckets()
            buckets = [bucket["Name"] for bucket in response.get("Buckets", [])]
            log.info(f"Found {len(buckets)} buckets: {', '.join(buckets)}")
            return buckets
        except ClientError as e:
            log.error(f"Error listing buckets: {e}")
            return []

    def upload_file(self, file_path, bucket_name, object_name=None):
        """
        Upload a file to a bucket.

        Args:
            file_path: Path to the file to upload
            bucket_name: Name of the bucket
            object_name: S3 object name (if None, uses file_path basename)

        Returns:
            bool: True if file was uploaded, False on error
        """
        # If object_name was not specified, use file_path basename
        if object_name is None:
            object_name = Path(file_path).name

        try:
            self.client.upload_file(file_path, bucket_name, object_name)
            log.debug(f"File {file_path} uploaded to {bucket_name}/{object_name}")
            return True
        except ClientError as e:
            log.error(f"Error uploading file {file_path}: {e}")
            return False

    def upload_directory(self, dir_path, bucket_name, prefix=""):
        """
        Upload all files from a directory to a bucket.

        Args:
            dir_path: Path to the directory to upload
            bucket_name: Name of the bucket
            prefix: Prefix to add to object names

        Returns:
            int: Number of files uploaded
        """
        dir_path = Path(dir_path)
        count = 0

        if not dir_path.is_dir():
            log.error(f"Error: {dir_path} is not a directory")
            return count

        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path from dir_path
                rel_path = file_path.relative_to(dir_path)
                # Create object name with prefix
                if prefix:
                    object_name = f"{prefix}/{rel_path}"
                else:
                    object_name = str(rel_path)

                if self.upload_file(str(file_path), bucket_name, object_name):
                    count += 1

        log.info(f"Uploaded {count} files to {bucket_name} from {dir_path}")
        return count

    def download_file(self, bucket_name, object_name, file_path=None):
        """
        Download a file from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name
            file_path: Local path to save the file (if None, uses object_name basename)

        Returns:
            bool: True if file was downloaded, False on error
        """
        # If file_path was not specified, use object_name basename
        if file_path is None:
            file_path = Path(object_name).name

        try:
            # Create directory if it doesn't exist
            os.makedirs(Path(file_path).parent, exist_ok=True)

            self.client.download_file(bucket_name, object_name, file_path)
            log.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")
            return True
        except ClientError as e:
            log.error(f"Error downloading {bucket_name}/{object_name}: {e}")
            return False

    def download_directory(self, bucket_name, prefix, local_dir=None):
        """
        Download all files with a prefix from a bucket.

        Args:
            bucket_name: Name of the bucket
            prefix: Prefix of objects to download
            local_dir: Local directory to save files (if None, uses current dir)

        Returns:
            int: Number of files downloaded
        """
        if local_dir is None:
            local_dir = "."

        local_dir = Path(local_dir)
        os.makedirs(local_dir, exist_ok=True)

        count = 0
        try:
            # List all objects with the prefix
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    object_name = obj["Key"]

                    # Calculate relative path from prefix
                    if prefix and object_name.startswith(prefix):
                        rel_path = object_name[len(prefix) :]
                        if rel_path.startswith("/"):
                            rel_path = rel_path[1:]
                    else:
                        rel_path = object_name

                    # Create local file path
                    file_path = local_dir / rel_path

                    if self.download_file(bucket_name, object_name, str(file_path)):
                        count += 1

            log.info(
                f"Downloaded {count} files from {bucket_name}/{prefix} to {local_dir}"
            )
            return count
        except ClientError as e:
            log.error(f"Error downloading from {bucket_name}/{prefix}: {e}")
            return count

    def list_objects(self, bucket_name, prefix="", recursive=True, max_items=None, print=True):
        """
        List objects in a bucket with an optional prefix.

        Args:
            bucket_name: Name of the bucket
            prefix: Prefix filter for objects
            recursive: If False, emulates directory listing with delimiters
            max_items: Maximum number of items to return
            print: If True, prints the list of objects

        Returns:
            list: List of object keys
        """
        try:
            paginator = self.client.get_paginator("list_objects_v2")

            # Set up pagination parameters
            pagination_config = {}
            if max_items:
                pagination_config["MaxItems"] = max_items

            # Set up operation parameters
            operation_params = {"Bucket": bucket_name, "Prefix": prefix}

            # If not recursive, use delimiter to emulate directory listing
            if not recursive:
                operation_params["Delimiter"] = "/"

            # Get pages of objects
            pages = paginator.paginate(
                **operation_params, PaginationConfig=pagination_config
            )

            objects = []

            for page in pages:
                # Add objects
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects.append(obj["Key"])

                # Add common prefixes (folders) if not recursive
                if not recursive and "CommonPrefixes" in page:
                    for prefix in page["CommonPrefixes"]:
                        objects.append(prefix["Prefix"])

            if print:
                log.info(f"Found {len(objects)} objects in {bucket_name}/{prefix}")
                for obj in objects:
                    log.info(f"- {obj}")

            return objects
        except ClientError as e:
            log.error(f"Error listing objects in {bucket_name}/{prefix}: {e}")
            return []

    def delete_object(self, bucket_name, object_name):
        """
        Delete an object from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name to delete

        Returns:
            bool: True if object was deleted, False on error
        """
        try:
            self.client.delete_object(Bucket=bucket_name, Key=object_name)
            log.info(f"Deleted {bucket_name}/{object_name}")
            return True
        except ClientError as e:
            log.error(f"Error deleting {bucket_name}/{object_name}: {e}")
            return False

    def delete_objects(self, bucket_name, object_names):
        """
        Delete multiple objects from a bucket.

        Args:
            bucket_name: Name of the bucket
            object_names: List of object names to delete

        Returns:
            int: Number of objects deleted
        """
        if not object_names:
            return 0

        try:
            # Create delete request
            objects = [{"Key": obj} for obj in object_names]
            response = self.client.delete_objects(
                Bucket=bucket_name, Delete={"Objects": objects}
            )

            deleted = len(response.get("Deleted", []))
            errors = len(response.get("Errors", []))

            log.info(f"Deleted {deleted} objects from {bucket_name}")
            if errors > 0:
                log.error(f"Failed to delete {errors} objects")

            return deleted
        except ClientError as e:
            log.error(f"Error deleting objects from {bucket_name}: {e}")
            return 0

    def delete_prefix(self, bucket_name, prefix):
        """
        Delete all objects with a specific prefix (like a folder).

        Args:
            bucket_name: Name of the bucket
            prefix: Prefix of objects to delete

        Returns:
            int: Number of objects deleted
        """
        try:
            # List all objects with the prefix
            objects = self.list_objects(bucket_name, prefix, print=False)

            # Delete the objects in batches
            count = 0
            batch_size = 1000  # S3 limits delete_objects to 1000 at a time

            for i in range(0, len(objects), batch_size):
                batch = objects[i : i + batch_size]
                count += self.delete_objects(bucket_name, batch)

            log.info(f"Deleted {count} objects from {bucket_name}/{prefix}")
            return count
        except ClientError as e:
            log.error(f"Error deleting prefix {bucket_name}/{prefix}: {e}")
            return 0

    def delete_all_objects(self, bucket_name):
        """
        Delete all objects in a bucket.

        Args:
            bucket_name: Name of the bucket

        Returns:
            int: Number of objects deleted
        """
        return self.delete_prefix(bucket_name, "")

    def split_directory_to_buckets(
        self, source_path, bucket_name, folder_names, split_folders=None
    ):
        """
        Split folders from a directory into separate folders in a bucket.

        Args:
            source_path: Path to the directory containing folders to split
            bucket_name: Name of the bucket to upload to
            folder_names: List of folder names to upload
            split_folders: Dictionary mapping folders to destination prefixes,
                           if None, splits into equal groups

        Returns:
            dict: Mapping of destination prefixes to lists of folders uploaded
        """
        source_path = Path(source_path)
        if not source_path.is_dir():
            log.error(f"Error: {source_path} is not a directory")
            return {}

        # Ensure bucket exists
        self.create_bucket(bucket_name)

        # Get folders in source directory that match requested folder names
        folders = []
        for folder_name in folder_names:
            folder_path = source_path / folder_name
            if folder_path.is_dir():
                folders.append(folder_name)
            else:
                log.warning(f"Warning: {folder_path} is not a directory, skipping")

        # If split_folders is None, create equal groups
        if split_folders is None:
            half = len(folders) // 2
            split_folders = {"1": folders[:half], "2": folders[half:]}

        result = {}

        # Upload each group of folders to the specified prefix
        for prefix, group_folders in split_folders.items():
            result[prefix] = []

            for folder in group_folders:
                if folder in folders:
                    folder_path = source_path / folder
                    # Upload the folder with the prefix
                    upload_prefix = f"{prefix}/{folder}"
                    count = self.upload_directory(
                        folder_path, bucket_name, upload_prefix
                    )
                    if count > 0:
                        result[prefix].append(folder)
                        log.info(f"Uploaded {folder} to {bucket_name}/{upload_prefix}")

        return result

    def copy_object(self, source_bucket, source_key, dest_bucket, dest_key=None):
        """
        Copy an object within or between buckets.

        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key (if None, uses source_key)

        Returns:
            bool: True if object was copied, False on error
        """
        if dest_key is None:
            dest_key = source_key

        try:
            copy_source = {"Bucket": source_bucket, "Key": source_key}

            self.client.copy_object(
                CopySource=copy_source, Bucket=dest_bucket, Key=dest_key
            )

            log.info(f"Copied {source_bucket}/{source_key} to {dest_bucket}/{dest_key}")
            return True
        except ClientError as e:
            log.error(f"Error copying {source_bucket}/{source_key}: {e}")
            return False

    def search_objects(self, bucket_name, pattern, prefix=""):
        """
        Search for objects in a bucket using a glob pattern.

        Args:
            bucket_name: Name of the bucket
            pattern: Glob pattern to match object keys against
            prefix: Optional prefix to limit search scope

        Returns:
            list: List of matching object keys
        """
        objects = self.list_objects(bucket_name, prefix)
        matches = [obj for obj in objects if fnmatch.fnmatch(obj, pattern)]

        log.info(
            f"Found {len(matches)} objects matching '{pattern}' in {bucket_name}/{prefix}"
        )
        for obj in matches:
            log.info(f"- {obj}")

        return matches
