# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import logging

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.ssh_helper as sh


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

    def __init__(self, collaborator_name=None, data_directory_path=None, workspace_path=None, container_id=None):
        """
        Initialize the Collaborator class
        Args:
            collaborator_name (str): Collaborator name
            data_directory_path (str): Data directory path
            workspace_path (str): Workspace path
            container_id (str): Container ID
        """
        self.name = collaborator_name
        self.collaborator_name = collaborator_name
        self.data_directory_path = data_directory_path
        self.workspace_path = workspace_path
        self.container_id = container_id

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
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, f"Generated a sign request for {self.collaborator_name}")

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e

    def create_collaborator(self):
        """
        Create the collaborator
        Returns:
            bool: True if successful, else False
        """
        try:
            cmd = f"fx collaborator create -n {self.collaborator_name} -d {self.data_directory_path}"
            error_msg = f"Failed to create {self.collaborator_name}"
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            fh.verify_cmd_output(
                output, return_code, error, error_msg,
                f"Created {self.collaborator_name} with the data directory {self.data_directory_path}"
            )

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
    
    def import_pki(self, agg_workspace_path):
        """
        Import and certify the CSR for the collaborator
        Args:
            agg_workspace_path (str): Workspace path of model owner or aggregator
        Returns:
            bool: True if successful, else False
        """
        try:
            zip_name = f"agg_to_col_{self.collaborator_name}_signed_cert.zip"
            signed_zip = os.path.join(agg_workspace_path, zip_name)
            cmd = f"fx collaborator certify --import {signed_zip}"
            error_msg = f"Failed to create {self.collaborator_name}"
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            fh.verify_cmd_output(
                output, return_code, error, error_msg,
                f"Successfully imported and certified the CSR for {self.collaborator_name} with zip path {signed_zip}"
            )

        except Exception as e:
            log.error(f"Failed to import and certify the CSR: {e}")
            raise e
        return True

    def start(self):
        """
        Start the collaborator
        Returns:
            str: Path to the log file
        """
        try:
            log.info(f"Starting {self.collaborator_name}")
            error_msg = f"Failed to start {self.collaborator_name}"
            res_file = os.path.join(self.workspace_path, f"{self.name}.log")
            fh.run_command(
                f"fx collaborator start -n {self.collaborator_name}",
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
                run_in_background=True,
                bg_file=res_file,
            )
            log.info(
                f"Started {self.name} and tracking the logs in {res_file}."
            )
        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return res_file

    def install_dependencies(self):
        """
        Install the dependencies for the collaborator
        Returns:
            bool: True if successful, else False
        """
        try:
            cmd = f"pip install -r requirements.txt"
            error_msg = f"Failed to install dependencies for {self.collaborator_name}"
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, f"Installed dependencies for {self.collaborator_name}")
        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def setup_col_docker_env(self, results_dir, workspace_template):
        """
        Setup the collaborator docker environment
        """
        try:
            container = dh.start_docker_container(
                container_name=self.collaborator_name,
                results_dir=results_dir,
                workspace_template=workspace_template,
            )
            self.container_id = container.id

            log.info(f"Setup of {self.collaborator_name} docker environment is complete")
        except Exception as e:
            log.error(f"Failed to setup {self.collaborator_name} docker environment: {e}")
            raise e
        
    def import_workspace(self, workspace_zip):
        """
        Import the workspace
        Args:
            workspace_zip (str): Path to the workspace zip file including the file name
        """
        try:
            cmd = f"fx workspace import --archive {workspace_zip}"
            error_msg = "Failed to export the workspace"
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=os.path.join(self.workspace_path, ".."), # Import the workspace to the parent directory
                print_output=True,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, f"Imported the workspace for {self.collaborator_name}")

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
