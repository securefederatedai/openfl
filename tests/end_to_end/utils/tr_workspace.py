# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import concurrent.futures
import logging
import os
from pathlib import Path

import tests.end_to_end.utils.data_helper as data_helper
import tests.end_to_end.utils.defaults as defaults
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.helper as helper
import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.models import aggregator as agg_model, model_owner as mo_model
import tests.end_to_end.utils.docker_helper as dh

log = logging.getLogger(__name__)

# Define a named tuple to store the objects for model owner, aggregator, and
# collaborators
federation_details = collections.namedtuple(
    "federation_details",
    "model_owner, aggregator, collaborators, workspace_path, local_bind_path, "
    "model_name",
)

def common_workspace_creation(request, eval_scope=False):
    """
    Common workspace creation function for task runner and dockerized workspace.

    Args:
        request (object): Pytest request object.
        eval_scope (bool, optional): If True, sets up the evaluation scope for a
        single round. Defaults to False.

    Returns:
        tuple: A tuple containing the workspace path, local bind path, aggregator
        domain name, model owner, plan path, and aggregator workspace path.
    """

    workspace_path, local_bind_path, agg_domain_name = (
        fh.federation_env_setup_and_validate(request, eval_scope)
    )

    agg_workspace_path = defaults.AGG_WORKSPACE_PATH.format(workspace_path)

    # For Flower App Pytorch, num of rounds must be 1
    if request.config.model_name.lower() == defaults.ModelName.FLOWER_APP_PYTORCH.value:
        if request.config.num_rounds != 1:
            raise ex.FlowerAppException(
                "Flower app with PyTorch only supports 1 round of training."
            )

    # Create model owner object and the workspace for the model
    # Workspace name will be same as the model name
    model_owner = mo_model.ModelOwner(
        request.config.model_name, request.config.log_memory_usage, workspace_path=agg_workspace_path
    )

    # Create workspace for given model name
    fh.create_persistent_store(model_owner.name, local_bind_path)

    model_owner.create_workspace()

    # Modify the plan
    plan_path = defaults.AGG_PLAN_PATH.format(local_bind_path)
    param_config = request.config

    initial_model_path = None
    if eval_scope:
        log.info(
            "Setting up evaluation scope, update the plan for 1 round and initial "
            "model to previous experiment best model"
        )
        initial_model_path = request.config.best_model_path

    model_owner.modify_plan(param_config, plan_path=plan_path)

    if hasattr(request.config, 'straggler_policy') and request.config.straggler_policy:
        model_owner.modify_straggler_policy(
            request.config.straggler_policy, plan_path=plan_path
        )

    return workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path


def create_tr_workspace(request, eval_scope=False):
    """
    Create a task runner workspace.

    Args:
        request (object): Pytest request object.
        eval_scope (bool, optional): If True, sets up the evaluation scope for a
        single round. Defaults to False.

    Returns:
        tuple : A named tuple containing the objects for model owner, aggregator,
        and collaborators.
    """
    if not request.config.model_name:
        raise ex.ModelNameException("Model name is not set in the request")

    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path = common_workspace_creation(request, eval_scope)

    # Initialize the plan
    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, extra_args=f"-i {initial_model_path}" if initial_model_path else ""
    )

    # Certify the workspace in case of TLS
    if request.config.use_tls:
        model_owner.certify_workspace()

    # Register the collaborators. It will also update plan/cols.yaml file with the collaborator names.
    model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        transport_protocol=request.config.transport_protocol,
        eval_scope=eval_scope,
        container_id=model_owner.container_id,  # None in case of native environment
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    # Export the workspace
    # By default the workspace will be exported to workspace.zip
    model_owner.export_workspace()

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()


    # In case of torch/histology_s3, we need to pass the data path, flag to calculate hash
    # and bucket mapping to the setup_collaborator function
    futures = [
        executor.submit(
            fh.setup_collaborator,
            index,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
            transport_protocol=request.config.transport_protocol,
        )
        for index in range(1, request.config.num_collaborators+1)
    ]

    collaborators = [f.result() for f in futures]

    # Data setup requires total no of collaborators, thus keeping the function call
    # outside of the loop
    if request.config.model_name.lower() in [defaults.ModelName.XGB_HIGGS.value, defaults.ModelName.FLOWER_APP_PYTORCH.value]:
        data_helper.setup_collaborator_data(collaborators, request.config.model_name, local_bind_path)

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)
        fh.import_pki_for_collaborators(collaborators)

    helper.remove_stale_processes(aggregator, collaborators)

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=request.config.model_name,
    )


def create_tr_workspace_gandlf(request, eval_scope=False):
    """
    Create a task runner workspace for Gandlf model.

    Args:
        request (object): Pytest request
    """
    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path = common_workspace_creation(request, eval_scope)

    home_dir = Path().home()
    results_path = os.path.join(home_dir, request.config.results_dir)

    # Raise exception if openfl does not contain gandlf folder.
    if not os.path.isdir(os.path.join(os.getcwd(), "gandlf")):
        raise Exception(
            "Folder 'gandlf' is not present in the current working directory. "
            "Please ensure that all the pre-requisites are met before running the test. "
            "Refer file .github/workflows/gandlf.yaml for the same."
        )

    # Check if valid.csv and train.csv are present in openfl folder
    if not os.path.exists(os.path.join(results_path, "valid.csv")) or not os.path.exists(
        os.path.join(results_path, "train.csv")
    ):
        raise ex.DataSetupException("Required data files for GanDLF are missing in the openfl folder")

    # Check if file config_segmentation.yaml is present in openfl folder
    gandlf_seg_file = os.path.join(results_path, "config_segmentation.yaml")
    if not os.path.exists(gandlf_seg_file):
        raise ex.GaNDLFConfigSegException(f"File {gandlf_seg_file} not available.")

    with open(gandlf_seg_file, 'r') as file:
        content = file.read()

    if not "num_channels" in content:
        raise ex.GaNDLFConfigSegException(f"File {gandlf_seg_file} must contain entry for num_channels.")

    # Create the objects for aggregator
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        transport_protocol=request.config.transport_protocol,
        eval_scope=eval_scope,
        container_id=model_owner.container_id,  # None in case of native environment
    )

    # Currently plan initialization internally checks data path in data.yaml
    # So we need to have data and modified data.yaml file in place before initializing the plan
    # Issue - https://github.com/securefederatedai/openfl/issues/73
    data_helper.download_gandlf_data(aggregator, local_bind_path, request.config.num_collaborators, results_path)

    # Initialize the plan
    extra_args = f"--gandlf_config {gandlf_seg_file}"
    extra_args += f" -i {initial_model_path}" if initial_model_path else ""

    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, extra_args=extra_args
    )

    # Update cols.yaml file with the collaborator names
    model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Certify the workspace, generate the sign request and certify the aggregator
    if request.config.use_tls:
        model_owner.certify_workspace()
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    # Export the workspace
    # By default the workspace will be exported to workspace.zip
    model_owner.export_workspace()

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    futures = [
        executor.submit(
            fh.setup_collaborator,
            index,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
            transport_protocol=request.config.transport_protocol,
        )
        for index in range(1, request.config.num_collaborators+1)
    ]
    collaborators = [f.result() for f in futures]

    data_helper.copy_gandlf_data_to_collaborators(aggregator, collaborators, local_bind_path)

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)
        fh.import_pki_for_collaborators(collaborators)

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=request.config.model_name,
    )


def create_tr_dws_workspace(request, eval_scope=False):
    """
    Run task runner experiment thru dockerized workspace.

    Args:
        request (object): Pytest request object.
        eval_scope (bool, optional): If True, sets up the evaluation scope for a
        single round. Defaults to False.

    Returns:
        tuple: A named tuple containing the objects for model owner, aggregator,
        and collaborators.
    """
    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path = common_workspace_creation(request, eval_scope)

    # Initialize the plan
    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, extra_args=f"-i {initial_model_path}" if initial_model_path else ""
    )

    # Create openfl image
    dh.build_docker_image(defaults.DEFAULT_OPENFL_IMAGE, defaults.DEFAULT_OPENFL_DOCKERFILE)

    # Command 'fx workspace dockerize --save ..' will use the workspace name for
    # image name which is 'workspace' in this case.
    model_owner.dockerize_workspace(defaults.DEFAULT_OPENFL_IMAGE)

    # Certify the workspace in case of TLS
    if request.config.use_tls:
        model_owner.certify_workspace()

    # Register the collaborators. It will also update plan/cols.yaml file with the collaborator names.
    model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        transport_protocol=request.config.transport_protocol,
        eval_scope=eval_scope,
        container_id=model_owner.container_id,  # None in case of native environment
    )

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = [
        executor.submit(
            fh.setup_collaborator,
            index,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
            transport_protocol=request.config.transport_protocol,
        )
        for index in range(1, request.config.num_collaborators + 1)
    ]
    collaborators = [f.result() for f in futures]

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)

    # Data setup requires total no of collaborators, thus keeping the function call
    # outside of the loop
    if request.config.model_name.lower() in [defaults.ModelName.XGB_HIGGS.value, defaults.ModelName.FLOWER_APP_PYTORCH.value]:
        data_helper.setup_collaborator_data(collaborators, request.config.model_name, local_bind_path)

    # Note: In case of multiple machines setup, scp the created tar for collaborators
    # to the other machine(s)
    fh.create_tarball_for_collaborators(
        collaborators, local_bind_path, use_tls=request.config.use_tls,
        add_data=True if request.config.model_name.lower() in [defaults.ModelName.XGB_HIGGS.value, defaults.ModelName.FLOWER_APP_PYTORCH.value] else False
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    local_agg_ws_path = defaults.AGG_WORKSPACE_PATH.format(local_bind_path)

    # Note: In case of multiple machines setup, scp this tar to the other machine(s)
    return_code, output, error = ssh.run_command(
        f"tar -cf cert_aggregator.tar plan cert save", work_dir=local_agg_ws_path
    )
    if return_code != 0:
        raise Exception(f"Failed to create tar for aggregator: {error}")

    # Note: In case of multiple machines setup, scp this workspace tar
    # to the other machine(s) so that docker load can load the image.
    model_owner.load_workspace(workspace_tar_name=f"{defaults.DFLT_WORKSPACE_NAME}.tar")

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=request.config.model_name,
    )
