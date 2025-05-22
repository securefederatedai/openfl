# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import tempfile

import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.helper as helper
import tests.end_to_end.utils.ssh_helper as ssh

log = logging.getLogger(__name__)


# Define the Collaborator class
class Collaborator():
    """
    Collaborator class to handle the collaborator operations.
    This includes (non-exhaustive list):
    1. Generating the sign request
    2. Creating the collaborator
    3. Importing and certifying the CSR
    4. Starting the collaborator
    """

    def __init__(self, collaborator_name, transport_protocol, data_directory_path=None, workspace_path=None, container_id=None):
        """
        Initialize the Collaborator class
        Args:
            collaborator_name (str): Collaborator name
            data_directory_path (str): Data directory path
            workspace_path (str): Workspace path
            container_id (str): Container ID
            transport_protocol (str): Transport protocol (default: "gRPC")
        """
        self.name = collaborator_name
        self.collaborator_name = collaborator_name
        self.data_directory_path = data_directory_path
        self.workspace_path = workspace_path
        self.container_id = container_id
        self.transport_protocol = transport_protocol
        self.res_file = None # Result file to track the logs
        self.start_process = None # Process associated with the aggregator start command

    def generate_sign_request(self):
        """
        Generate a sign request for the collaborator
        Returns:
            bool: True if successful, else False
        """
        try:
            log.info(f"Generating a sign request for {self.collaborator_name}")
            cmd = f"fx collaborator generate-cert-request -n {self.collaborator_name}"
            error_msg = "Failed to generate the sign request"
            return_code, output, error = helper.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            helper.verify_cmd_output(output, return_code, error, error_msg, f"Generated a sign request for {self.collaborator_name}")

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def create_collaborator(self):
        """
        Create the collaborator
        Returns:
            bool: True if successful, else False
        """
        try:
            cmd = f"fx collaborator create -n {self.collaborator_name} -d {self.data_directory_path}"
            error_msg = f"Failed to create {self.collaborator_name}"
            return_code, output, error = helper.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            helper.verify_cmd_output(
                output, return_code, error, error_msg,
                f"Created {self.collaborator_name} with the data directory {self.data_directory_path}"
            )

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e

    def import_pki(self, zip_name, with_docker=False):
        """
        Import and certify the CSR for the collaborator
        Args:
            zip_name (str): Zip file name
            with_docker (bool): Flag specific to dockerized workspace scenario. Default is False.
        Returns:
            bool: True if successful, else False
        """
        # Assumption - zip file is present in the collaborator workspace
        try:
            cmd = f"fx collaborator certify --import {zip_name}"
            error_msg = f"Failed to import and certify the CSR for {self.collaborator_name}"
            return_code, output, error = helper.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path if not with_docker else "",
                with_docker=with_docker,
            )
            helper.verify_cmd_output(
                output, return_code, error, error_msg,
                f"Successfully imported and certified the CSR for {self.collaborator_name} with zip {zip_name}"
            )

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def start(self):
        """
        Start the collaborator
        Returns:
            bool: True if successful, else raise exception
        """
        try:
            log.info(f"Starting {self.collaborator_name}")
            error_msg = f"Failed to start {self.collaborator_name}"

            # Note: LOG_FILE does not take absolute path, hence using relative path
            log_file = os.path.join("logs", f"{self.collaborator_name}.log")
            self.res_file = os.path.join(self.workspace_path, log_file)

            command = ["fx", "collaborator", "start", "-n", self.collaborator_name]
            log.info(f"Command for {self.name}: {command}")

            # Set the log file path for the collaborator process
            env = os.environ.copy()
            env["LOG_FILE"] = log_file

            # open file in append mode, so that restarting scenarios can be handled
            bg_file = open(os.path.join(tempfile.mkdtemp(), "tmp.log"), "a", buffering=1)
            self.start_process = ssh.run_command_background(
                cmd=command,
                work_dir=self.workspace_path,
                redirect_to_file=bg_file,
                check_sleep=30,
                env=env
            )

            log.info(
                f"Started {self.name} and tracking the logs in {self.res_file}."
            )
        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def kill_process(self):
        """
        Kill the process of the collaborator and wait for it to finish
        """
        try:
            if self.start_process:
                self.start_process.kill()
                self.start_process.wait()
                self.start_process = None
            else:
                log.warning(f"No process found for {self.collaborator_name}")

        except Exception as e:
            log.error(f"Failed to kill the process: {e}")
            raise ex.ProcessKillException(f"Failed to kill the process: {e}")

    def import_workspace(self):
        """
        Import the workspace
        Args:
            agg_workspace_path (str): Workspace path of model owner or aggregator
        """
        try:
            # Assumption - workspace.zip is present in the collaborator workspace
            cmd = f"fx workspace import --archive {self.workspace_path}/workspace.zip"
            error_msg = "Failed to import the workspace"
            return_code, output, error = helper.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=os.path.join(self.workspace_path, ".."), # Import the workspace to the parent directory
            )
            helper.verify_cmd_output(output, return_code, error, error_msg, f"Imported the workspace for {self.collaborator_name}")

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e

    def modify_data_file(self, data_file, index):
        """
        Modify the data.yaml file for the model
        Args:
            data_file (str): Path to the data file including the file name
        Returns:
            bool: True if successful, else False
        """
        try:
            log.info("Data setup completed successfully. Modifying the data.yaml file..")

            with open(data_file, "w") as file:
                file.write(f"{self.collaborator_name},data/{index}")

        except Exception as e:
            log.error(f"Failed to modify the data file: {e}")
            raise ex.DataSetupException(f"Failed to modify the data file: {e}")

        return True

    def ping_aggregator(self):
        """
        Ping the aggregator to check if it is running
        Returns:
            bool: True if successful, else False
        """
        try:
            log.info(f"Pinging from {self.collaborator_name} to aggregator")
            # Note: LOG_FILE does not take absolute path, hence using relative path
            log_file = os.path.join("logs", f"{self.collaborator_name}.log")
            self.res_file = os.path.join(self.workspace_path, log_file)
            # Set the log file path for the collaborator process
            env = os.environ.copy()
            env["LOG_FILE"] = log_file
            command = ["fx", "collaborator", "ping", "-n", self.collaborator_name]
            error_msg = f"Failed to ping from {self.collaborator_name}"
            # run in background to avoid blocking the main thread
            bg_file = open(os.path.join(tempfile.mkdtemp(), "tmp.log"), "a", buffering=1)
            self.start_process = ssh.run_command_background(
                cmd=command,
                work_dir=self.workspace_path,
                redirect_to_file=bg_file,
                check_sleep=30,
                env=env
            )
            log.info(
                f"Pinged from {self.name} and tracking the logs in {self.res_file}."
            )
        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def calculate_hash(self):
        """
        Calculate the hash of the data directory and store in hash.txt file
        Returns:
            bool: True if successful, else False
        """
        try:
            log.info(f"Calculating hash for {self.collaborator_name}")
            cmd = f"fx collaborator calchash --data_path {self.data_directory_path}"
            error_msg = "Failed to calculate hash"
            return_code, output, error = helper.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            helper.verify_cmd_output(output, return_code, error, error_msg, f"Calculated hash for {self.collaborator_name}")

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True
