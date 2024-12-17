# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
import pytest
import os
import shutil
import random
from metaflow import Step, Flow

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

def test_exclude_flow(fx_local_federated_workflow):
    flflow = TestFlowExclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

def test_include_exclude_flow(fx_local_federated_workflow):
    flflow = TestFlowIncludeExclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

def test_include_flow(fx_local_federated_workflow):
    flflow = TestFlowInclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

def test_internal_loop(fx_local_federated_workflow):
    model = None
    optimizer = None

    flflow = TestFlowInternalLoop(model, optimizer, 5, checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

    # Flow Test Begins
    expected_flow_steps = [
        "join",
        "internal_loop",
        "agg_model_mean",
        "collab_model_update",
        "local_model_mean",
        "start",
        "end",
    ]  # List to verify expected steps

    steps_present_in_cli, missing_steps_in_cli, extra_steps_in_cli = wf_helper.validate_flow(
            flflow, expected_flow_steps
        )  # Function to validate the internal flow

    assert len(steps_present_in_cli) == len(expected_flow_steps), "Number of steps fetched from Datastore through CLI do not match the Expected steps provided"
    assert len(missing_steps_in_cli) == 0, f"Following steps missing from Datastore: {missing_steps_in_cli}"
    assert len(extra_steps_in_cli) == 0, f"Following steps are extra in Datastore: {extra_steps_in_cli}"
    assert flflow.end_count == 1, "End function called more than one time"

    log.info("\n **** Summary of internal flow testing ****\n"
             "No issues found and below are the tests that ran successfully\n"
             "1. Number of training completed is equal to training rounds\n"
             "2. Cli steps and Expected steps are matching\n"
             "3. Number of tasks are aligned with number of rounds and number of collaborators\n"
             "4. End function executed one time")


@pytest.mark.parametrize("fx_local_federated_workflow", [("init_collaborator_private_attr_index", "int", None )], indirect=True)
def test_reference_flow(fx_local_federated_workflow):
    flflow = TestFlowReference(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()


def test_reference_include_flow(fx_local_federated_workflow):
    flflow = TestFlowReferenceWithInclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()


def test_reference_exclude_flow(fx_local_federated_workflow):
    flflow = TestFlowReferenceWithExclude(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

@pytest.mark.parametrize("fx_local_federated_workflow", [("init_collaborator_private_attr_name", "str", None )], indirect=True)
def test_subset_collaborators(fx_local_federated_workflow):
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
        # We now convert names to lowercase
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


def test_private_attr_wo_callable(fx_local_federated_workflow_prvt_attr):
    flflow = TestFlowPrivateAttributesWoCallable(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow_prvt_attr.runtime
    for i in range(5):
        print(f"Starting round {i}...")
        flflow.run()


@pytest.mark.parametrize("fx_local_federated_workflow", [("init_collaborate_pvt_attr_np", "int", "init_agg_pvt_attr_np" )], indirect=True)
def test_private_attributes(fx_local_federated_workflow):
    flflow = TestFlowPrivateAttributes(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow.runtime
    flflow.run()

@pytest.mark.parametrize("fx_local_federated_workflow_prvt_attr", [("init_collaborate_pvt_attr_np", "int", "init_agg_pvt_attr_np" )], indirect=True)
def test_private_attr_both(fx_local_federated_workflow_prvt_attr):
    flflow = TestFlowPrivateAttributesBoth(checkpoint=True)
    flflow.runtime = fx_local_federated_workflow_prvt_attr.runtime
    for i in range(5):
        print(f"Starting round {i}...")
        flflow.run()
