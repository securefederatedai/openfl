# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import logging

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.ssh_helper as ssh

log = logging.getLogger(__name__)


# Define the ModelOwner class
class ModelOwner():
    """
    ModelOwner class to handle the model related operations.
    Note: Aggregator can also act as a model owner.
    This includes (non-exhaustive list):
    1. Creating the workspace - to create a workspace using given workspace and model names.
    2. Modifying based on input params provided and initializing the plan.
    3. Certifying the workspace and setting up the PKI.
    4. Importing and exporting the workspace etc.
    """
    def __init__(self, model_name, log_memory_usage, container_id=None, workspace_path=None):
        """
        Initialize the ModelOwner class
        Args:
            model_name (str): Model name
            log_memory_usage (bool): Memory Log flag
            container_id (str, Optional): Container ID
            workspace_path (str, Optional): Workspace path
        """
        self.workspace_name = model_name # keeping workspace name same as model name for simplicity
        self.model_name = model_name
        self.name = "aggregator"
        self.aggregator = None
        self.collaborators = []
        self.workspace_path = workspace_path
        self.num_collaborators = constants.NUM_COLLABORATORS
        self.rounds_to_train = constants.NUM_ROUNDS
        self.log_memory_usage = log_memory_usage
        self.container_id = container_id

    def create_workspace(self):
        """
        Create the workspace for the model
        """
        try:
            log.info(f"Creating workspace for model {self.model_name} at the path: {self.workspace_path}")
            error_msg = "Failed to create the workspace"

            ws_path = self.workspace_path

            return_code, output, error = fh.run_command(
                f"fx workspace create --prefix {ws_path} --template {self.model_name}",
                workspace_path="", # No workspace path required for this command
                error_msg=error_msg,
                container_id=self.container_id,
            )
            fh.verify_cmd_output(
                output,
                return_code,
                error,
                error_msg, f"Created the workspace {self.workspace_name} for the {self.model_name} model",
                raise_exception=True
            )

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e

    def get_workspace_path(self, results_dir, workspace_name):
        """
        Get the workspace path
        Args:
            results_dir (str): Results directory path
            workspace_name (str): Workspace name
        Returns:
            str: Path to the workspace
        """
        workspace_path = os.path.join(results_dir, workspace_name)
        log.info(f"Workspace path: {workspace_path}")
        if os.path.exists(workspace_path):
            self.workspace_path = workspace_path
            log.info(f"Workspace path: {self.workspace_path}")
        else:
            log.error(f"Workspace {workspace_name} does not exist in {results_dir}")
            raise FileNotFoundError(f"Workspace {workspace_name} does not exist in {results_dir}")
        return self.workspace_path

    def certify_collaborator(self, collaborator_name, zip_name):
        """
        Sign the CSR for the collaborator
        Args:
            collaborator_name (str): Collaborator name
            zip_name (str): Zip file name
        """
        # Assumption - CSR is already created by the collaborator and copied to the aggregator workspace
        try:
            cmd = f"fx collaborator certify --request-pkg {zip_name} -s"
            error_msg = f"Failed to sign the CSR {zip_name}"
            return_code, output, error = fh.run_command(
                cmd,
                workspace_path=self.workspace_path,
                error_msg=error_msg,
                container_id=self.container_id,
            )

            fh.verify_cmd_output(
                output,
                return_code,
                error,
                error_msg,
                f"Successfully signed the CSR {zip_name}"
            )

        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return True

    def modify_plan(self, param_config, plan_path):
        """
        Modify the plan to train the model
        Args:
            param_config (object): Config object containing various params to be modified
            plan_path (str): Path to the plan file
        """
        # Copy the cols.yaml file from remote machine to local machine for docker environment
        plan_file = os.path.join(plan_path, "plan.yaml")

        # Open the file and modify the entries
        self.rounds_to_train = param_config.num_rounds if param_config.num_rounds else self.rounds_to_train
        self.num_collaborators = param_config.num_collaborators if param_config.num_collaborators else self.num_collaborators

        try:
            with open(plan_file) as fp:
                data = yaml.load(fp, Loader=yaml.FullLoader)

            # NOTE: If more parameters need to be modified, add them here
            data["aggregator"]["settings"]["rounds_to_train"] = int(self.rounds_to_train)
            # Memory Leak related
            data["aggregator"]["settings"]["log_memory_usage"] = self.log_memory_usage
            data["collaborator"]["settings"]["log_memory_usage"] = self.log_memory_usage

            data["data_loader"]["settings"]["collaborator_count"] = int(self.num_collaborators)
            data["network"]["settings"]["require_client_auth"] = param_config.require_client_auth
            data["network"]["settings"]["use_tls"] = param_config.use_tls

            with open(plan_file, "w+") as write_file:
                yaml.dump(data, write_file)

            log.info(f"Modified the plan with provided parameters.")
        except Exception as e:
            log.error(f"Failed to modify the plan: {e}")
            raise ex.PlanModificationException(f"Failed to modify the plan: {e}")

    def initialize_plan(self, agg_domain_name):
        """
        Initialize the plan
        Args:
            agg_domain_name (str): Aggregator domain name
        """
        try:
            log.info("Initializing the plan. It will take some time to complete..")
            cmd = f"fx plan initialize -a {agg_domain_name}"
            error_msg="Failed to initialize the plan"
            return_code, output, error = fh.run_command(
                cmd,
                workspace_path=self.workspace_path,
                error_msg=error_msg,
                container_id=self.container_id,
            )
            fh.verify_cmd_output(
                output,
                return_code,
                error,
                error_msg,
                f"Initialized the plan for the workspace {self.workspace_name}"
            )

        except Exception as e:
            raise ex.PlanInitializationException(f"{error_msg}: {e}")

    def certify_workspace(self):
        """
        Certify the workspace
        Returns:
            bool: True if successful, else False
        """
        try:
            log.info("Certifying the workspace..")
            cmd = f"fx workspace certify"
            error_msg = "Failed to certify the workspace"
            return_code, output, error = fh.run_command(
                cmd,
                workspace_path=self.workspace_path,
                error_msg=error_msg,
                container_id=self.container_id,
            )
            fh.verify_cmd_output(
                output,
                return_code,
                error,
                error_msg,
                f"Certified the workspace {self.workspace_name}"
            )

        except Exception as e:
            raise ex.WorkspaceCertificationException(f"{error_msg}: {e}")

    def dockerize_workspace(self):
        """
        Dockerize the workspace. It internally uses workspace name as the image name
        """
        log.info("Dockerizing the workspace. It will take some time to complete..")
        try:
            if not os.getenv("GITHUB_REPOSITORY") or not os.getenv("GITHUB_BRANCH"):
                repo, branch = ssh.get_git_repo_and_branch()
            else:
                repo = os.getenv("GITHUB_REPOSITORY")
                branch = os.getenv("GITHUB_BRANCH")

            cmd = f"fx workspace dockerize --save --revision {repo}@{branch}"
            error_msg = "Failed to dockerize the workspace"
            return_code, output, error = fh.run_command(
                cmd,
                workspace_path=self.workspace_path,
                error_msg=error_msg,
                container_id=self.container_id,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, "Workspace dockerized successfully")

        except Exception as e:
            raise ex.WorkspaceDockerizationException(f"{error_msg}: {e}")

    def load_workspace(self, workspace_tar_name):
        """
        Load the workspace
        """
        log.info("Loading the workspace..")
        try:
            return_code, output, error = ssh.run_command(f"docker load -i {workspace_tar_name}", work_dir=self.workspace_path)
            if return_code != 0:
                raise Exception(f"Failed to load the workspace: {error}")

        except Exception as e:
            raise ex.WorkspaceLoadException(f"Error loading workspace: {e}")

    def register_collaborators(self, plan_path, num_collaborators=None):
        """
        Register the collaborators
        Args:
            plan_path (str): Path to the plan file
            num_collaborators (int, Optional): Number of collaborators
        Returns:
            bool: True if successful, else False
        """
        log.info(f"Registering the collaborators..")
        cols_file = os.path.join(plan_path, "cols.yaml")
        self.num_collaborators = num_collaborators if num_collaborators else self.num_collaborators

        try:
            # Open the file and add the entries from scratch.
            # This way even if there is a mismatch with some models having it blank
            # and others having values, it will be consistent
            with open(cols_file, "r", encoding="utf-8") as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)

            doc["collaborators"] = []  # Create empty list

            for i in range(num_collaborators):
                col_name = "collaborator" + str(i+1)
                doc["collaborators"].append(col_name)
                with open(cols_file, "w", encoding="utf-8") as f:
                    yaml.dump(doc, f)

            log.info(
                f"Successfully registered collaborators in {cols_file}"
            )
        except Exception as e:
            raise ex.CollaboratorRegistrationException(f"Failed to register the collaborators: {e}")

    def certify_aggregator(self, agg_domain_name):
        """
        Certify the aggregator request
        Args:
            agg_domain_name (str): Aggregator domain name
        Returns:
            bool: True if successful, else False
        """
        log.info(f"Certify the aggregator request")
        try:
            cmd = f"fx aggregator certify --silent --fqdn {agg_domain_name}"
            error_msg = "Failed to certify the aggregator request"
            return_code, output, error = fh.run_command(
                cmd,
                workspace_path=self.workspace_path,
                error_msg=error_msg,
                container_id=self.container_id,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, "CA signed the request from aggregator")

        except Exception as e:
            raise ex.AggregatorCertificationException(f"{error_msg}: {e}")

    def export_workspace(self):
        """
        Export the workspace
        """
        try:
            cmd = "fx workspace export"
            error_msg = "Failed to export the workspace"
            return_code, output, error = fh.run_command(
                cmd,
                workspace_path=self.workspace_path,
                error_msg=error_msg,
                container_id=self.container_id,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, "Workspace exported successfully")

        except Exception as e:
            raise ex.WorkspaceExportException(f"{error_msg}: {e}")
