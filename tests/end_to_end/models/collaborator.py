# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import logging

import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.federation_helper as fh

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
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path if not with_docker else "",
                with_docker=with_docker,
            )
            fh.verify_cmd_output(
                output, return_code, error, error_msg,
                f"Successfully imported and certified the CSR for {self.collaborator_name} with zip {zip_name}"
            )

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def start(self, res_file, with_docker=False):
        """
        Start the collaborator
        Args:
            res_file (str): Result file to track the logs
            with_docker (bool): Flag to run the collaborator inside a docker container
        Returns:
            str: Path to the log file
        """
        try:
            log.info(f"Starting {self.collaborator_name}")
            res_file = res_file if not with_docker else os.path.basename(res_file)
            error_msg = f"Failed to start {self.collaborator_name}"
            fh.run_command(
                f"fx collaborator start -n {self.collaborator_name}",
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path if not with_docker else "",
                run_in_background=True,
                bg_file=res_file,
                with_docker=with_docker
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
        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def setup_col_docker_env(self, workspace_path, local_bind_path):
        """
        Setup the collaborator docker environment
        Args:
            workspace_path (str): Workspace path
            local_bind_path (str): Local bind path
        """
        try:
            container = dh.start_docker_container(
                container_name=self.collaborator_name,
                workspace_path=workspace_path,
                local_bind_path=local_bind_path,
            )
            self.container_id = container.id

            log.info(f"Setup of {self.collaborator_name} docker environment is complete")
        except Exception as e:
            log.error(f"Failed to setup {self.collaborator_name} docker environment: {e}")
            raise e

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
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=os.path.join(self.workspace_path, ".."), # Import the workspace to the parent directory
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, f"Imported the workspace for {self.collaborator_name}")

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
