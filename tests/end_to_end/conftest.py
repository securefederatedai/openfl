# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import collections
import os
import shutil
import xml.etree.ElementTree as ET
import logging
import concurrent.futures

from tests.end_to_end.utils.logger import configure_logging
from tests.end_to_end.utils.logger import logger as log
from tests.end_to_end.utils.conftest_helper import parse_arguments
import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.federation_helper as fh
from tests.end_to_end.models import aggregator as agg_model, model_owner as mo_model

# Define a named tuple to store the objects for model owner, aggregator, and collaborators
federation_fixture = collections.namedtuple(
    "federation_fixture",
    "model_owner, aggregator, collaborators, workspace_path, local_bind_path",
)

def pytest_addoption(parser):
    """
    Add custom command line options to the pytest parser.
    Args:
        parser: pytest parser object
    """
    parser.addini("results_dir", "Directory to store test results", default="results")
    parser.addini("log_level", "Logging level", default="DEBUG")
    parser.addoption("--num_collaborators")
    parser.addoption("--num_rounds")
    parser.addoption("--model_name")
    parser.addoption("--disable_client_auth", action="store_true")
    parser.addoption("--disable_tls", action="store_true")
    parser.addoption("--log_memory_usage", action="store_true")


def pytest_configure(config):
    """
    Configure the pytest plugin.
    Args:
        config: pytest config object
    """
    # Declare some global variables
    args = parse_arguments()
    # Use the model name from the test case name if not provided as a command line argument
    config.model_name = args.model_name
    config.num_collaborators = args.num_collaborators
    config.num_rounds = args.num_rounds
    config.require_client_auth = not args.disable_client_auth
    config.use_tls = not args.disable_tls
    config.log_memory_usage = args.log_memory_usage
    config.results_dir = config.getini("results_dir")


@pytest.fixture(scope="session", autouse=True)
def setup_logging(pytestconfig):
    """
    Setup logging for the test session.
    Args:
        pytestconfig: pytest config object
    Returns:
        logger: logger object
    """
    tmp_results_dir = pytestconfig.getini("results_dir")
    log_level = pytestconfig.getini("log_level")

    results_dir = os.path.join(os.getenv("HOME"), tmp_results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Setup a global logger to ensure logging works before any test-specific logs are set
    configure_logging(f"{results_dir}/deployment.log", log_level)
    return logging.getLogger()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture the result of setup, call, and teardown phases.
    This avoids duplicate entries for Pass/Fail in the XML report.
    """
    outcome = yield
    report = outcome.get_result()

    # Retrieve the custom test_id marker if it exists
    test_id_marker = item.get_closest_marker("test_id")
    outcome_mapping = {"passed": "Pass", "failed": "Fail"}
    report_when_mapping = {"setup": "Setup", "call": "Test", "teardown": "Teardown"}
    final_outcome = outcome_mapping.get(report.outcome, report.outcome)
    report_phase = report_when_mapping.get(report.when, report.when)

    # Modify nodeid if test_id is provided and append outcome and phase
    if test_id_marker:
        test_id = test_id_marker.args[0]
        report.nodeid = (
            f"{report.nodeid} [{test_id}] [outcome: {final_outcome}] [phase: {report_phase}]"
        )

    # Initialize XML structure if not already initialized
    if not hasattr(item.config, "_xml_report"):
        item.config._xml_report = ET.Element(
            "testsuite",
            {
                "name": "pytest",
                "errors": "0",
                "failures": "0",
                "skipped": "0",
                "tests": "0",
                "time": "0",
                "timestamp": "",
                "hostname": "",
            },
        )

    # Store the result of each phase (setup/call/teardown)
    if not hasattr(item, "_results"):
        item._results = {}

    # Save the outcome and other details per phase
    item._results[report.when] = {
        "outcome": final_outcome,
        "longrepr": report.longrepr,
        "duration": report.duration,
    }
    # Log failures
    if report.when == "call" and report.failed:
        logger = logging.getLogger()
        logger.error(f"Test {report.nodeid} failed: {call.excinfo.value}")

    # Only create the XML element after the teardown phase
    if report.when == "teardown" and not hasattr(item, "_xml_created"):
        item._xml_created = True  # Ensure XML creation happens only once

        # Determine final outcome based on the worst phase result
        if "call" in item._results:
            final_outcome = item._results["call"]["outcome"]
        elif "setup" in item._results:
            final_outcome = item._results["setup"]["outcome"]
        else:
            final_outcome = "skipped"

        # Create the <testcase> XML element
        testcase = ET.SubElement(
            item.config._xml_report,
            "testcase",
            {
                "classname": item.module.__name__,
                "name": item.name,
                "time": str(sum(result["duration"] for result in item._results.values())),
            },
        )

        # Add <failure> or <skipped> tags based on the final outcome
        if final_outcome == "Fail":
            failure_message = item._results.get("call", {}).get(
                "longrepr", item._results.get("setup", {}).get("longrepr", "Unknown Error")
            )
            failure = ET.SubElement(
                testcase,
                "error",
                {
                    "message": str(failure_message),
                },
            )
            failure.text = str(failure_message)
        elif final_outcome == "skipped":
            skipped_message = item._results.get("setup", {}).get("longrepr", "Skipped")
            skipped = ET.SubElement(
                testcase,
                "skipped",
                {
                    "message": str(skipped_message),
                },
            )
            skipped.text = str(skipped_message)

        # Update the testsuite summary statistics
        tests = int(item.config._xml_report.attrib["tests"]) + 1
        item.config._xml_report.attrib["tests"] = str(tests)
        if final_outcome == "Fail":
            failures = int(item.config._xml_report.attrib["failures"]) + 1
            item.config._xml_report.attrib["failures"] = str(failures)
        elif final_outcome == "skipped":
            skipped = int(item.config._xml_report.attrib["skipped"]) + 1
            item.config._xml_report.attrib["skipped"] = str(skipped)


def pytest_sessionfinish(session, exitstatus):
    """
    Operations to be performed after the test session is finished.
    More functionalities to be added in this function in future.
    """
    cache_dir = os.path.join(session.config.rootdir, ".pytest_cache")
    log.debug(f"\nClearing .pytest_cache directory at {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=False)
        log.debug(f"Cleared .pytest_cache directory at {cache_dir}")


def pytest_configure(config):
    """
    Configure the pytest plugin.
    Args:
        config: pytest config object
    """
    # Declare some global variables
    args = parse_arguments()
    # Use the model name from the test case name if not provided as a command line argument
    config.model_name = args.model_name
    config.num_collaborators = args.num_collaborators
    config.num_rounds = args.num_rounds
    config.require_client_auth = not args.disable_client_auth
    config.use_tls = not args.disable_tls
    config.log_memory_usage = args.log_memory_usage
    config.results_dir = config.getini("results_dir")


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
