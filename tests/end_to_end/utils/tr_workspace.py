# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import concurrent.futures
import logging

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.federation_helper as fh
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

    model_name, workspace_path, local_bind_path, agg_domain_name = (
        fh.federation_env_setup_and_validate(request, eval_scope)
    )

    agg_workspace_path = constants.AGG_WORKSPACE_PATH.format(workspace_path)

    # Create model owner object and the workspace for the model
    # Workspace name will be same as the model name
    model_owner = mo_model.ModelOwner(
        model_name, request.config.log_memory_usage, workspace_path=agg_workspace_path
    )

    # Create workspace for given model name
    fh.create_persistent_store(model_owner.name, local_bind_path)

    model_owner.create_workspace()
    fh.add_local_workspace_permission(local_bind_path)

    # Modify the plan
    plan_path = constants.AGG_PLAN_PATH.format(local_bind_path)
    param_config = request.config

    initial_model_path = None
    if eval_scope:
        log.info(
            "Setting up evaluation scope, update the plan for 1 round and initial "
            "model to previous experiment best model"
        )
        param_config.num_rounds = 1
        initial_model_path = request.config.best_model_path

    model_owner.modify_plan(param_config, plan_path=plan_path, eval_scope=eval_scope)

    if hasattr(request.config, 'straggler_policy') and request.config.straggler_policy:
        model_owner.modify_straggler_policy(
            request.config.straggler_policy, plan_path=plan_path
        )

    # Initialize the plan
    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, initial_model_path=initial_model_path
    )

    # Return the federation fixture
    return workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path


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
    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path = common_workspace_creation(request, eval_scope)
    model_name = request.config.model_name
    # Certify the workspace in case of TLS
    # Register the collaborators in case of non-TLS
    if request.config.use_tls:
        model_owner.certify_workspace()
    else:
        model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
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

    futures = [
        executor.submit(
            fh.setup_collaborator,
            index,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
        )
        for index in range(1, request.config.num_collaborators+1)
    ]
    collaborators = [f.result() for f in futures]

    # Data setup requires total no of collaborators, thus keeping the function call
    # outside of the loop
    if model_name.lower() == "xgb_higgs":
        fh.setup_collaborator_data(collaborators, model_name, local_bind_path)

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)
        fh.import_pki_for_collaborators(collaborators, local_bind_path)

    fh.remove_stale_processes(request.config.num_collaborators)

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=model_name,
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
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path = common_workspace_creation(request, eval_scope)
    model_name = request.config.model_name

    # Create openfl image
    dh.build_docker_image(constants.DEFAULT_OPENFL_IMAGE, constants.DEFAULT_OPENFL_DOCKERFILE)

    # Command 'fx workspace dockerize --save ..' will use the workspace name for
    # image name which is 'workspace' in this case.
    model_owner.dockerize_workspace(constants.DEFAULT_OPENFL_IMAGE)
    image_name = constants.DFLT_DOCKERIZE_IMAGE_NAME

    # Certify the workspace in case of TLS
    # Register the collaborators in case of non-TLS
    if request.config.use_tls:
        model_owner.certify_workspace()
    else:
        model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
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
        )
        for index in range(1, request.config.num_collaborators + 1)
    ]
    collaborators = [f.result() for f in futures]

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)

    # Data setup requires total no of collaborators, thus keeping the function call
    # outside of the loop
    if model_name.lower() == "xgb_higgs":
        fh.setup_collaborator_data(collaborators, model_name, local_bind_path)

    # Note: In case of multiple machines setup, scp the created tar for collaborators
    # to the other machine(s)
    fh.create_tarball_for_collaborators(
        collaborators, local_bind_path, use_tls=request.config.use_tls,
        add_data=True if model_name.lower() == "xgb_higgs" else False
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)

    # Note: In case of multiple machines setup, scp this tar to the other machine(s)
    return_code, output, error = ssh.run_command(
        f"tar -cf cert_aggregator.tar plan cert save", work_dir=local_agg_ws_path
    )
    if return_code != 0:
        raise Exception(f"Failed to create tar for aggregator: {error}")

    # Note: In case of multiple machines setup, scp this workspace tar
    # to the other machine(s) so that docker load can load the image.
    model_owner.load_workspace(workspace_tar_name=f"{image_name}.tar")

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=model_name,
    )
