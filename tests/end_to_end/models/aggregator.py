# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import tempfile

import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.helper as helper
import tests.end_to_end.utils.ssh_helper as ssh


log = logging.getLogger(__name__)

# Define the Aggregator class
class Aggregator():
    """
    Aggregator class to handle the aggregator operations.
    This includes (non-exhaustive list):
    1. Generating the sign request
    2. Starting the aggregator
    """

    def __init__(self, agg_domain_name, workspace_path, transport_protocol, eval_scope=False, container_id=None):
        """
        Initialize the Aggregator class
        Args:
            agg_domain_name (str): Aggregator domain name
            workspace_path (str): Workspace path
            container_id (str): Container ID
            eval_scope (bool, optional): Scope of aggregator is evaluation. Default is False.
            transport_protocol (str): Transport protocol (default: "gRPC")
        """
        self.name = "aggregator"
        self.agg_domain_name = agg_domain_name
        self.workspace_path = workspace_path
        self.eval_scope = eval_scope
        self.container_id = container_id
        self.transport_protocol = transport_protocol
        self.tensor_db_file = os.path.join(self.workspace_path, "local_state", "tensor.db")
        self.res_file = None # Result file to track the logs
        self.start_process = None # Process associated with the aggregator start command

    def generate_sign_request(self):
        """
        Generate a sign request for the aggregator
        """
        try:
            cmd = f"fx aggregator generate-cert-request --fqdn {self.agg_domain_name}"
            error_msg = "Failed to generate the sign request"
            return_code, output, error = helper.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            helper.verify_cmd_output(output, return_code, error, error_msg, f"Generated a sign request for {self.name}")

        except Exception as e:
            raise ex.CSRGenerationException(f"Failed to generate sign request for {self.name}: {e}")

    def start(self):
        """
        Start the aggregator
        Returns:
            bool: True if successful, else raise exception
        """
        try:
            log.info(f"Starting {self.name}")
            error_msg = "Failed to start the aggregator"

            # Note: LOG_FILE does not take absolute path, hence using relative path
            log_file = os.path.join("logs", "aggregator.log")
            self.res_file = os.path.join(self.workspace_path, log_file)

            command = ["fx", "aggregator", "start"]
            if self.eval_scope:
                command.append("--task_group")
                command.append("evaluation")
            log.info(f"Command for {self.name}: {command}")

            # Set the log file path for the aggregator process
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
        Kill the process of the aggregator and wait for it to finish
        """
        try:
            if self.start_process:
                self.start_process.kill()
                self.start_process.wait()
                self.start_process = None
            else:
                log.warning("No process found for aggregator")
        except Exception as e:
            log.error(f"Failed to kill the process: {e}")
            raise ex.ProcessKillException(f"Failed to kill the process: {e}")

    def modify_data_file(self, data_file, col_name, index):
        """
        Modify the data.yaml file for the model
        Args:
            data_file (str): Path to the data file including the file name
        Returns:
            bool: True if successful, else False
        """
        try:
            log.info("Data setup completed successfully. Modifying the data.yaml file..")

            with open(data_file, "a") as file:
                file.write(f"{col_name},data/{index}\n")

        except Exception as e:
            log.error(f"Failed to modify the data file: {e}")
            raise ex.DataSetupException(f"Failed to modify the data file: {e}")

        return True
