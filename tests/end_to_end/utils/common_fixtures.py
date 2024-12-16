# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import collections
import concurrent.futures
import os

import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.federation_helper as fh
from tests.end_to_end.models import aggregator as agg_model, model_owner as mo_model


# Define a named tuple to store the objects for model owner, aggregator, and collaborators
federation_fixture = collections.namedtuple(
    "federation_fixture",
    "model_owner, aggregator, collaborators, workspace_path, local_bind_path",
)


@pytest.fixture(scope="function")
def fx_federation(request):
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
    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    test_env, model_name, workspace_path, local_bind_path, agg_domain_name = fh.federation_env_setup_and_validate(request)
    agg_workspace_path = os.path.join(workspace_path, "aggregator", "workspace")

    # Create model owner object and the workspace for the model
    # Workspace name will be same as the model name
    model_owner = mo_model.ModelOwner(model_name, request.config.log_memory_usage, workspace_path=agg_workspace_path)

    # Create workspace for given model name
    fh.create_persistent_store(model_owner.name, local_bind_path)

    # Start the docker container for aggregator in case of docker environment
    if test_env == "docker":
        container = dh.start_docker_container(
            container_name="aggregator",
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
        )
        model_owner.container_id = container.id

    model_owner.create_workspace()
    fh.add_local_workspace_permission(local_bind_path)

    # Modify the plan
    plan_path = os.path.join(local_bind_path, "aggregator", "workspace", "plan")
    model_owner.modify_plan(
        plan_path=plan_path,
        new_rounds=request.config.num_rounds,
        num_collaborators=request.config.num_collaborators,
        disable_client_auth=not request.config.require_client_auth,
        disable_tls=not request.config.use_tls,
    )

    # Certify the workspace in case of TLS
    # Register the collaborators in case of non-TLS
    if request.config.use_tls:
        model_owner.certify_workspace()
    else:
        model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Initialize the plan
    model_owner.initialize_plan(agg_domain_name=agg_domain_name)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        container_id=model_owner.container_id, # None in case of non-docker environment
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

    # Return the federation fixture
    return federation_fixture(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
    )
