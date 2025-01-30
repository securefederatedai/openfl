# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import collections
import numpy as np

from openfl.experimental.workflow.interface import Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime

# Define a named tuple to store the objects for model owner, aggregator, and collaborators
workflow_local_fixture = collections.namedtuple(
    "workflow_local_fixture",
    "aggregator, collaborators, runtime",
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
    # Inline import
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
    # Inline import
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
