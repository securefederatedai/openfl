# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import collections
import concurrent.futures
import logging

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.docker_helper as dh
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
    test_env = fh.get_test_env_from_markers(request)

    if test_env not in ["task_runner_docker", "task_runner_basic"]:
        raise ValueError("Use fx_federation_tr_dws for this test environment: dockerized_ws")
    
    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    model_name, workspace_path, local_bind_path, agg_domain_name = fh.federation_env_setup_and_validate(request)

    agg_workspace_path = constants.AGG_WORKSPACE_PATH.format(workspace_path)

    # Create model owner object and the workspace for the model
    # Workspace name will be same as the model name
    model_owner = mo_model.ModelOwner(model_name, request.config.log_memory_usage, workspace_path=agg_workspace_path)

    # Create workspace for given model name
    fh.create_persistent_store(model_owner.name, local_bind_path)

    # Start the docker container for aggregator in case of docker environment
    if test_env == "task_runner_docker":
        container = dh.start_docker_container(
            container_name="aggregator",
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
        )
        model_owner.container_id = container.id

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
        container_id=model_owner.container_id, # None in case of non-docker environment
    )

    # Generate the sign request and certify the aggregator in case of TLS
    # Skip this step in case of dockerized workspace
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
    if fh.get_test_env_from_markers(request) != "dockerized_ws":
        raise ValueError("Use fx_federation_tr_dws for this test environment: dockerized_ws")

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    model_name, workspace_path, local_bind_path, agg_domain_name = fh.federation_env_setup_and_validate(request)
 
    agg_workspace_path = constants.AGG_WORKSPACE_PATH.format(workspace_path)

    # Create model owner object and the workspace for the model
    # Workspace name will be same as the model name
    model_owner = mo_model.ModelOwner(model_name, request.config.log_memory_usage, workspace_path=agg_workspace_path)

    # Create workspace for given model name
    fh.create_persistent_store(model_owner.name, local_bind_path)

    model_owner.create_workspace()
    fh.add_local_workspace_permission(local_bind_path)

    # Modify the plan
    plan_path = constants.AGG_PLAN_PATH.format(local_bind_path)
    model_owner.modify_plan(param_config=request.config, plan_path=plan_path)

    # Initialize the plan
    model_owner.initialize_plan(agg_domain_name=agg_domain_name)

    model_owner.dockerize_workspace()

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
        container_id=model_owner.container_id, # None in case of non-docker environment
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

    fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)

    fh.create_tarball_for_collaborators(collaborators, local_bind_path)

    # Generate the sign request and certify the aggregator in case of TLS
    # Skip this step in case of dockerized workspace
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)
        local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)
        return_code, output, error = ssh.run_command(f"tar -cf cert_agg.tar plan cert save", work_dir=local_agg_ws_path)
        if return_code != 0:
            raise Exception(f"Failed to create tar for aggregator: {error}")

    # When no name is provided 'fx workspace dockerize --save ..' will use the last folder name
    # which is workspace in this case for tar and image name.
    image_name = "workspace"
    model_owner.load_workspace(workspace_tar_name=f"{image_name}.tar")

    futures = [
        executor.submit(
            dh.start_docker_container,
            container_name=participant.name,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
            image=image_name,
            mount_mapping=["cert_agg.tar:/certs.tar"] if participant.name == "aggregator" else [f"cert_col_{participant.name}.tar:/certs.tar"],
        )
        for participant in collaborators + [aggregator]
    ]
    results = [f.result() for f in futures]
    log.info(f"Result of starting docker containers: {results}")

    # Return the federation fixture
    return federation_fixture(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
    )
