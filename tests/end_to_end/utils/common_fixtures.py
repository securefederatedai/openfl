# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import collections
import concurrent.futures
import logging
import numpy as np

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

workflow_local_fixture = collections.namedtuple(
    "workflow_local_fixture",
    "aggregator, collaborators, runtime",
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


@pytest.fixture(scope="function")
def fx_local_federated_workflow(request):
    """
    Fixture to set up a local federated workflow for testing.
    This fixture initializes an `Aggregator` and sets up a list of collaborators
    based on the number specified in the test configuration. It also configures
    a `LocalRuntime` with the aggregator, collaborators, and an optional backend
    if specified in the test configuration.
    Args:
        request (FixtureRequest): The pytest request object that provides access
                                to the test configuration.
    Yields:
        LocalRuntime: An instance of `LocalRuntime` configured with the aggregator,
                    collaborators, and backend.
    """
    # Import is done inline because Task Runner does not support importing below openfl packages
    from openfl.experimental.workflow.interface import Aggregator, Collaborator
    from openfl.experimental.workflow.runtime import LocalRuntime
    from tests.end_to_end.utils.wf_helper import (
        init_collaborator_private_attr_index,
        init_collaborator_private_attr_name,
        init_collaborate_pvt_attr_np,
        init_agg_pvt_attr_np
    )
    collab_callback_func = request.param[0] if hasattr(request, 'param') and request.param else None
    collab_value = request.param[1] if hasattr(request, 'param') and request.param else None
    agg_callback_func = request.param[2] if hasattr(request, 'param') and request.param else None

    # Get the callback functions from the locals using string
    collab_callback_func_name = locals()[collab_callback_func] if collab_callback_func else None
    agg_callback_func_name = locals()[agg_callback_func] if agg_callback_func else None
    collaborators_list = []

    if agg_callback_func_name:
        aggregator = Aggregator( name="agg",
                                private_attributes_callable=agg_callback_func_name)
    else:
        aggregator = Aggregator()

    # Setup collaborators
    for i in range(request.config.num_collaborators):
        func_var = i if collab_value == "int" else f"collaborator{i}" if collab_value == "str" else None
        collaborators_list.append(
            Collaborator(
                name=f"collaborator{i}",
                private_attributes_callable=collab_callback_func_name,
                param = func_var
            )
        )

    backend = request.config.backend if hasattr(request.config, 'backend') else None
    if backend:
        local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators_list, backend=backend)
    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators_list)

    # Return the federation fixture
    return workflow_local_fixture(
        aggregator=aggregator,
        collaborators=collaborators_list,
        runtime=local_runtime,
    )


@pytest.fixture(scope="function")
def fx_local_federated_workflow_prvt_attr(request):
    """
    Fixture to set up a local federated workflow for testing.
    This fixture initializes an `Aggregator` and sets up a list of collaborators
    based on the number specified in the test configuration. It also configures
    a `LocalRuntime` with the aggregator, collaborators, and an optional backend
    if specified in the test configuration.
    Args:
        request (FixtureRequest): The pytest request object that provides access
                                to the test configuration.
    Yields:
        LocalRuntime: An instance of `LocalRuntime` configured with the aggregator,
                    collaborators, and backend.
    """
    # Import is done inline because Task Runner does not support importing below openfl packages
    from openfl.experimental.workflow.interface import Aggregator, Collaborator
    from openfl.experimental.workflow.runtime import LocalRuntime
    from tests.end_to_end.utils.wf_helper import (
        init_collaborator_private_attr_index,
        init_collaborator_private_attr_name,
        init_collaborate_pvt_attr_np,
        init_agg_pvt_attr_np
    )
    collab_callback_func = request.param[0] if hasattr(request, 'param') and request.param else None
    collab_value = request.param[1] if hasattr(request, 'param') and request.param else None
    agg_callback_func = request.param[2] if hasattr(request, 'param') and request.param else None

    # Get the callback functions from the locals using string
    collab_callback_func_name = locals()[collab_callback_func] if collab_callback_func else None
    agg_callback_func_name = locals()[agg_callback_func] if agg_callback_func else None
    collaborators_list = []

    # Setup aggregator
    if agg_callback_func_name:
        aggregator = Aggregator(name="agg",
                                private_attributes_callable=agg_callback_func_name)
    else:
        aggregator = Aggregator()

    aggregator.private_attributes = {
        "test_loader_pvt": np.random.rand(10, 28, 28)  # Random data
    }
    # Setup collaborators
    for i in range(request.config.num_collaborators):
        func_var = i if collab_value == "int" else f"collaborator{i}" if collab_value == "str" else None
        collab = Collaborator(
                name=f"collaborator{i}",
                private_attributes_callable=collab_callback_func_name,
                param = func_var
            )
        collab.private_attributes = {
            "train_loader_pvt": np.random.rand(i * 50, 28, 28),
            "test_loader_pvt": np.random.rand(i * 10, 28, 28),
        }
        collaborators_list.append(collab)

    backend = request.config.backend if hasattr(request.config, 'backend') else None
    if backend:
        local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators_list, backend=backend)
    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators_list)

    # Return the federation fixture
    return workflow_local_fixture(
        aggregator=aggregator,
        collaborators=collaborators_list,
        runtime=local_runtime,
    )
