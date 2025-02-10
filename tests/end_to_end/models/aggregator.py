# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.federation_helper as fh


log = logging.getLogger(__name__)

# Define the Aggregator class
class Aggregator():
    """
    Aggregator class to handle the aggregator operations.
    This includes (non-exhaustive list):
    1. Generating the sign request
    2. Starting the aggregator
    """

    def __init__(self, agg_domain_name, workspace_path, eval_scope=False, container_id=None):
        """
        Initialize the Aggregator class
        Args:
            agg_domain_name (str): Aggregator domain name
            workspace_path (str): Workspace path
            container_id (str): Container ID
            eval_scope (bool, optional): Scope of aggregator is evaluation. Default is False.
        """
        self.name = "aggregator"
        self.agg_domain_name = agg_domain_name
        self.workspace_path = workspace_path
        self.eval_scope = eval_scope
        self.container_id = container_id
        self.tensor_db_file = os.path.join(self.workspace_path, "local_state", "tensor.db")
        self.res_file = None # Result file to track the logs

    def generate_sign_request(self):
        """
        Generate a sign request for the aggregator
        """
        try:
            cmd = f"fx aggregator generate-cert-request --fqdn {self.agg_domain_name}"
            error_msg = "Failed to generate the sign request"
            return_code, output, error = fh.run_command(
                cmd,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
            )
            fh.verify_cmd_output(output, return_code, error, error_msg, f"Generated a sign request for {self.name}")

        except Exception as e:
            raise ex.CSRGenerationException(f"Failed to generate sign request for {self.name}: {e}")

    def start(self, res_file):
        """
        Start the aggregator
        Args:
            res_file (str): Result file to track the logs
        Returns:
            str: Path to the log file
        """
        try:
            log.info(f"Starting {self.name}")
            error_msg = "Failed to start the aggregator"
            command = constants.AGG_START_CMD
            if self.eval_scope:
                command = f"{command} --task_group evaluation"
            fh.run_command(
                command=command,
                error_msg=error_msg,
                container_id=self.container_id,
                workspace_path=self.workspace_path,
                run_in_background=True,
                bg_file=res_file,
            )
            log.info(
                f"Started {self.name} and tracking the logs in {res_file}."
            )
            self.res_file = res_file
        except Exception as e:
            log.error(f"{error_msg}: {e}")
            raise e
        return res_file
