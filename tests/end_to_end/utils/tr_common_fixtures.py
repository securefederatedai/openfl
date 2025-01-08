# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import collections
import concurrent.futures
import logging

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.models import aggregator as agg_model, model_owner as mo_model

log = logging.getLogger(__name__)


# Define a named tuple to store the objects for model owner, aggregator, and collaborators
federation_fixture = collections.namedtuple(
    "federation_fixture",
    "model_owner, aggregator, collaborators, workspace_path, local_bind_path",
)


@pytest.fixture(scope="function")
def fx_federation_tr(request):
    """
    Fixture for federation. This fixture is used to create the model owner, aggregator, and collaborators.
    It also creates workspace.
    Assumption: OpenFL workspace is present for the model being tested.
    Args:
        request: pytest request object. Model name is passed as a parameter to the fixture from test cases.
    Returns:
        federation_fixture: Named tuple containing the objects for model owner, aggregator, and collaborators

    Note: As this is a function level fixture, thus no import is required at test level.
    """
    request.config.test_env = "task_runner_basic"

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    model_name, workspace_path, local_bind_path, agg_domain_name = (
        fh.federation_env_setup_and_validate(request)
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
    model_owner.modify_plan(param_config=request.config, plan_path=plan_path)

    # Initialize the plan
    model_owner.initialize_plan(agg_domain_name=agg_domain_name)

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
        container_id=model_owner.container_id,  # None in case of non-docker environment
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    # Export the workspace
    # By default the workspace will be exported to workspace.zip
    model_owner.export_workspace()

    futures = [
        executor.submit(
            fh.setup_collaborator,
            count=i,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
        )
        for i in range(request.config.num_collaborators)
    ]
    collaborators = [f.result() for f in futures]

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)
        fh.import_pki_for_collaborators(collaborators, local_bind_path)

    # Return the federation fixture
    return federation_fixture(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
    )


@pytest.fixture(scope="function")
def fx_federation_tr_dws(request):
    """
    Fixture for federation in case of dockerized workspace. This fixture is used to create the model owner, aggregator, and collaborators.
    It also creates workspace.
    Assumption: OpenFL workspace is present for the model being tested.
    Args:
        request: pytest request object. Model name is passed as a parameter to the fixture from test cases.
    Returns:
        federation_fixture: Named tuple containing the objects for model owner, aggregator, and collaborators

    Note: As this is a function level fixture, thus no import is required at test level.
    """
    request.config.test_env = "task_runner_dockerized_ws"

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    model_name, workspace_path, local_bind_path, agg_domain_name = (
        fh.federation_env_setup_and_validate(request)
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
    model_owner.modify_plan(param_config=request.config, plan_path=plan_path)

    # Initialize the plan
    model_owner.initialize_plan(agg_domain_name=agg_domain_name)

    # Command 'fx workspace dockerize --save ..' will use the workspace name for image name
    # which is 'workspace' in this case.
    model_owner.dockerize_workspace()
    image_name = "workspace"

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
        container_id=model_owner.container_id,  # None in case of non-docker environment
    )

    futures = [
        executor.submit(
            fh.setup_collaborator,
            count=i,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
        )
        for i in range(request.config.num_collaborators)
    ]
    collaborators = [f.result() for f in futures]

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)

    # Note: In case of multiple machines setup, scp the created tar for collaborators to the other machine(s)
    fh.create_tarball_for_collaborators(
        collaborators, local_bind_path, use_tls=request.config.use_tls
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)

    # Note: In case of multiple machines setup, scp this tar to the other machine(s)
    return_code, output, error = ssh.run_command(
        f"tar -cf cert_agg.tar plan cert save", work_dir=local_agg_ws_path
    )
    if return_code != 0:
        raise Exception(f"Failed to create tar for aggregator: {error}")

    # Note: In case of multiple machines setup, scp this workspace tar
    # to the other machine(s) so that docker load can load the image.
    model_owner.load_workspace(workspace_tar_name=f"{image_name}.tar")

    fh.start_docker_containers_for_dws(
        participants=[aggregator] + collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        image_name=image_name,
    )

    # Return the federation fixture
    return federation_fixture(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
    )
