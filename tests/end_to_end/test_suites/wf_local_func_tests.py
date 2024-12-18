# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import pytest
import os
import shutil
import random
from metaflow import Step

from tests.end_to_end.utils.common_fixtures import fx_local_federated_workflow, fx_local_federated_workflow_prvt_attr
from tests.end_to_end.workflow.exclude_flow import TestFlowExclude
from tests.end_to_end.workflow.include_exclude_flow import TestFlowIncludeExclude
from tests.end_to_end.workflow.include_flow import TestFlowInclude
from tests.end_to_end.workflow.internal_loop import TestFlowInternalLoop
from tests.end_to_end.workflow.reference_flow import TestFlowReference
from tests.end_to_end.workflow.reference_include_flow import TestFlowReferenceWithInclude
from tests.end_to_end.workflow.reference_exclude import TestFlowReferenceWithExclude
from tests.end_to_end.workflow.subset_flow import TestFlowSubsetCollaborators
from tests.end_to_end.workflow.private_attr_wo_callable import TestFlowPrivateAttributesWoCallable
from tests.end_to_end.workflow.private_attributes_flow import TestFlowPrivateAttributes
from tests.end_to_end.workflow.private_attr_both import TestFlowPrivateAttributesBoth

from tests.end_to_end.utils import wf_helper as wf_helper

log = logging.getLogger(__name__)

def test_exclude_flow(request, fx_local_federated_workflow):
    """
    Test if variable is excluded, variables not show in next step
    and all other variables will be visible to next step
    """
    log.info("Starting test_exclude_flow")
    flflow = TestFlowExclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_exclude_flow")


def test_include_exclude_flow(request, fx_local_federated_workflow):
    """
    Test variables which are excluded will not show up in next step
    Test variables which are included will show up in next step
    """
    log.info("Starting test_include_exclude_flow")
    flflow = TestFlowIncludeExclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_include_exclude_flow")


def test_include_flow(request, fx_local_federated_workflow):
    """
    Test if variable is included, variables will show up in next step
    All other variables will not show up
    """
    log.info("Starting test_include_flow")
    flflow = TestFlowInclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_include_flow")


def test_internal_loop(request, fx_local_federated_workflow):
    """
    Verify that through internal loop, rounds to train is set
    """
    log.info("Starting test_internal_loop")
    model = None
    optimizer = None

    flflow = TestFlowInternalLoop(model, optimizer, request.config.num_rounds, checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

    expected_flow_steps = [
        "join",
        "internal_loop",
        "agg_model_mean",
        "collab_model_update",
        "local_model_mean",
        "start",
        "end",
    ]

    steps_present_in_cli, missing_steps_in_cli, extra_steps_in_cli = wf_helper.validate_flow(
            flflow, expected_flow_steps
        )

    assert len(steps_present_in_cli) == len(expected_flow_steps), "Number of steps fetched from Datastore through CLI do not match the Expected steps provided"
    assert len(missing_steps_in_cli) == 0, f"Following steps missing from Datastore: {missing_steps_in_cli}"
    assert len(extra_steps_in_cli) == 0, f"Following steps are extra in Datastore: {extra_steps_in_cli}"
    assert flflow.end_count == 1, "End function called more than one time"

    log.info("\n  Summary of internal flow testing \n"
             "No issues found and below are the tests that ran successfully\n"
             "1. Number of training completed is equal to training rounds\n"
             "2. CLI steps and Expected steps are matching\n"
             "3. Number of tasks are aligned with number of rounds and number of collaborators\n"
             "4. End function executed one time")
    log.info("Successfully ended test_internal_loop")


@pytest.mark.parametrize("fx_local_federated_workflow", [("init_collaborator_private_attr_index", "int", None )], indirect=True)
def test_reference_flow(request, fx_local_federated_workflow):
    """
    Test reference variables matched through out the flow
    """
    log.info("Starting test_reference_flow")
    flflow = TestFlowReference(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_reference_flow")


def test_reference_include_flow(request, fx_local_federated_workflow):
    """
    Test reference variables matched if included else not
    """
    log.info("Starting test_reference_include_flow")
    flflow = TestFlowReferenceWithInclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_reference_include_flow")


def test_reference_exclude_flow(request, fx_local_federated_workflow):
    """
    Test reference variables matched if not excluded
    """
    log.info("Starting test_reference_exclude_flow")
    flflow = TestFlowReferenceWithExclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_reference_exclude_flow")


@pytest.mark.parametrize("fx_local_federated_workflow", [("init_collaborator_private_attr_name", "str", None )], indirect=True)
def test_subset_collaborators(request, fx_local_federated_workflow):
    """
    Test the subset of collaborators in a federated workflow.

    Parameters:
        request (FixtureRequest): The request fixture provides information about the requesting test function.
        fx_local_federated_workflow (Fixture): The fixture for the local federated workflow.

    Tests:
        - Ensure the test starts and ends correctly.
        - Verify the number of collaborators matches the expected subset.
        - Check that the flow runs for each subset collaborator.
    """
    log.info("Starting test_subset_collaborators")
    collaborators = fx_local_federated_workflow.collaborators

    random_ints = random.sample(range(1, len(collaborators) + 1), len(collaborators))

    collaborators = fx_local_federated_workflow.runtime.collaborators
    for round_num in range(len(collaborators)):
        log.info(f"Starting round {round_num}...")

        if os.path.exists(".metaflow"):
            shutil.rmtree(".metaflow")

        flflow = TestFlowSubsetCollaborators(checkpoint=True, random_ints=random_ints)
        flflow.runtime = fx_local_federated_workflow.runtime
        flflow.run()
        subset_collaborators = flflow.subset_collaborators
        collaborators_ran = flflow.collaborators_ran
        random_ints = flflow.random_ints
        random_ints.remove(len(subset_collaborators))

        step = Step(
            f"TestFlowSubsetCollaborators/{flflow._run_id}/"
            + "test_valid_collaborators"
        )

        assert len(list(step)) == len(subset_collaborators), (
                f"...Flow only ran for {len(list(step))} "
                + f"instead of the {len(subset_collaborators)} expected "
                + f"collaborators- Testcase Failed."
            )
        log.info(
            f"Found {len(list(step))} tasks for each of the "
            + f"{len(subset_collaborators)} collaborators"
        )
        log.info(f'subset_collaborators = {subset_collaborators}')
        log.info(f'collaborators_ran = {collaborators_ran}')
        for collaborator_name in subset_collaborators:
            assert collaborator_name in collaborators_ran, (
                f"...Flow did not execute for "
                + f"collaborator {collaborator_name}"
                + f" - Testcase Failed."
            )

    log.info(
        f"Testing FederatedFlow - Ending test for validating "
        + f"the subset of collaborators.")
    log.info("Successfully ended test_subset_collaborators")


def test_private_attr_wo_callable(request, fx_local_federated_workflow_prvt_attr):
    """
    Set private attribute without callable function i.e through direct assignment
    """
    log.info("Starting test_private_attr_wo_callable")
    flflow = TestFlowPrivateAttributesWoCallable(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow_prvt_attr.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_private_attr_wo_callable")


@pytest.mark.parametrize("fx_local_federated_workflow", [("init_collaborate_pvt_attr_np", "int", "init_agg_pvt_attr_np" )], indirect=True)
def test_private_attributes(request, fx_local_federated_workflow):
    """
    Set private attribute through callable function
    """
    log.info("Starting test_private_attributes")
    flflow = TestFlowPrivateAttributes(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    for i in range(request.config.num_rounds):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_private_attributes")


@pytest.mark.parametrize("fx_local_federated_workflow_prvt_attr", [("init_collaborate_pvt_attr_np", "int", "init_agg_pvt_attr_np" )], indirect=True)
def test_private_attr_both(request, fx_local_federated_workflow_prvt_attr):
    """
    Set private attribute through callable function and direct assignment
    """
    log.info("Starting test_private_attr_both")
    flflow = TestFlowPrivateAttributesBoth(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow_prvt_attr.runtime
    for i in range(5):
        log.info(f"Starting round {i}...")
        flflow.run()
    log.info("Successfully ended test_private_attr_both")
