import pytest
import os
import json
import shutil
import xml.etree.ElementTree as ET
import logging
from utils.logger import configure_logging
from utils.logger import logger as log
from utils.argparse_helper import parse_arguments, get_default_repo_dir


def pytest_addoption(parser):
    parser.addini("results_dir", "Directory to store test results", default="results")
    parser.addini("log_level", "Logging level", default="DEBUG")
    parser.addoption("--num_collaborators", action="store", type=int, default=2, help="Number of collaborators")
    parser.addoption("--num_rounds", action="store", type=int, default=5, help="Number of rounds to train")
    parser.addoption("--model_name", action="store", type=str, default="torch_cnn_mnist", help="Model name")


@pytest.fixture(scope="session", autouse=True)
def setup_logging(pytestconfig):
    results_dir = pytestconfig.getini("results_dir")
    log_level = pytestconfig.getini("log_level")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Setup a global logger to ensure logging works before any test-specific logs are set
    configure_logging(os.path.join(results_dir, 'deployment.log'), log_level)
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
        report.nodeid = f"{report.nodeid} [{test_id}] [outcome: {final_outcome}] [phase: {report_phase}]"

    # Initialize XML structure if not already initialized
    if not hasattr(item.config, '_xml_report'):
        item.config._xml_report = ET.Element('testsuite', {
            'name': 'pytest',
            'errors': '0',
            'failures': '0',
            'skipped': '0',
            'tests': '0',
            'time': '0',
            'timestamp': '',
            'hostname': ''
        })

    # Store the result of each phase (setup/call/teardown)
    if not hasattr(item, '_results'):
        item._results = {}

    # Save the outcome and other details per phase
    item._results[report.when] = {
        'outcome': final_outcome,
        'longrepr': report.longrepr,
        'duration': report.duration,
    }
    # Log failures
    if report.when == 'call' and report.failed:
        logger = logging.getLogger()
        logger.error(f"Test {report.nodeid} failed: {call.excinfo.value}")

    # Only create the XML element after the teardown phase
    if report.when == 'teardown' and not hasattr(item, '_xml_created'):
        item._xml_created = True  # Ensure XML creation happens only once

        # Determine final outcome based on the worst phase result
        if 'call' in item._results:
            final_outcome = item._results['call']['outcome']
        elif 'setup' in item._results:
            final_outcome = item._results['setup']['outcome']
        else:
            final_outcome = 'skipped'

        # Create the <testcase> XML element
        testcase = ET.SubElement(
            item.config._xml_report, 'testcase', {
                'classname': item.module.__name__,
                'name': item.name,
                'time': str(sum(result['duration'] for result in item._results.values())),
            }
        )

        # Add <failure> or <skipped> tags based on the final outcome
        if final_outcome == 'Fail':
            failure_message = item._results.get('call', {}).get('longrepr', item._results.get('setup', {}).get('longrepr', 'Unknown Error'))
            failure = ET.SubElement(testcase, 'error', {
                'message': str(failure_message),
            })
            failure.text = str(failure_message)
        elif final_outcome == 'skipped':
            skipped_message = item._results.get('setup', {}).get('longrepr', 'Skipped')
            skipped = ET.SubElement(testcase, 'skipped', {
                'message': str(skipped_message),
            })
            skipped.text = str(skipped_message)

        # Update the testsuite summary statistics
        tests = int(item.config._xml_report.attrib['tests']) + 1
        item.config._xml_report.attrib['tests'] = str(tests)
        if final_outcome == 'Fail':
            failures = int(item.config._xml_report.attrib['failures']) + 1
            item.config._xml_report.attrib['failures'] = str(failures)
        elif final_outcome == 'skipped':
            skipped = int(item.config._xml_report.attrib['skipped']) + 1
            item.config._xml_report.attrib['skipped'] = str(skipped)

def pytest_sessionfinish(session, exitstatus):
    # Clear the .pytest_cache directory after the test session is finished
    cache_dir = os.path.join(session.config.rootdir, '.pytest_cache')
    print(f"Clearing .pytest_cache directory at {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=False)
        print(f"Cleared .pytest_cache directory at {cache_dir}")
        
# @pytest.fixture(scope="session")
# def global_config(pytestconfig):
#     args = parse_arguments()
#     config = {
#         "deploy_repo_path": pytestconfig.getoption("--deploy-repo-path") or args.deploy_repo_path,
#         "results_dir": pytestconfig.getoption("--results-dir") or args.results_dir or pytestconfig.getini(
#             "results_dir"),
#         "repo_dir": pytestconfig.getoption("--repo-dir") or args.repo_dir or get_default_repo_dir(),
#         "log_level": pytestconfig.getini("log_level"),
#         "num_collaborators": pytestconfig.getoption("--num-collaborators") or args.num_collaborators,
#         "test_mode": pytestconfig.getoption("--test-mode") or args.test_mode,
#         "browser_type": pytestconfig.getoption("--browser-type") or args.browser_type,
#         "keep_deployment": pytestconfig.getoption("--keep-deployment")
#     }
#     return config


# @pytest.fixture(autouse=True)
# def manage_logs(request, global_config):
#     """Set log file name same as test name and results_dir to include module name"""
#     results_dir = global_config['results_dir']
#     log_level = global_config['log_level']
#     suite_name = request.node.fspath.purebasename
#     test_name = request.node.name
#     module_results_dir = os.path.join(results_dir, suite_name)
#     os.makedirs(module_results_dir, exist_ok=True)
#     log_file = os.path.join(module_results_dir, f"{test_name}.log")

#     # Clear existing handlers, if any
#     logger = logging.getLogger()
#     while logger.handlers:
#         logger.handlers.pop()

#     # Configure logging for the specific test
#     configure_logging(log_file, log_level)
#     global_config['test_results_dir'] = module_results_dir
#     global_config['test_name'] = test_name


# @pytest.fixture(scope="function")
# def setup_ui(federation, global_config):
#     browser_type = global_config['browser_type']
#     setup_ui = UISetup(federation, browser_type)
#     yield setup_ui
#     setup_ui.quit_driver()


# @pytest.fixture(scope="module")
# def federation(request, global_config):
#     # load the existing deployment if it exists
#     cached_deployment = utils_helper.load_deployment()
#     # Deploy federation
#     num_collaborators = global_config['num_collaborators']

#     deployment = init_deployment(num_collaborators, global_config)
#     init_participants(deployment)
    
#     if cached_deployment and deployment.check_az_deployment_exists():
#         # load the deployment information from the cache
#         deployment.governor.token = cached_deployment['governor_token']
#         deployment.aggregator.token = cached_deployment['aggregator_token']
#         deployment.model_owner.token = cached_deployment['model_owner_token']
#         # Initialize tokens for all collaborators
#         for i, collaborator in enumerate(deployment.collaborators):
#             collaborator.token = cached_deployment['collaborator_tokens'][i]

#         log.info("Loaded deployment information from cache")
#         yield deployment
#         return
    
#     utils_helper.create_ssh_key_pairs()
#     if not deploy_federation(deployment, global_config['test_mode']):
#         log.error("Deployment failed")
#         delete_federation(deployment)
#         log.info("Federation deleted")
#         raise Exception("Deployment failed")

#     log.info("Deploying federation completed")
    
#     # save the deployment information
#     utils_helper.save_deployment(deployment)

#     yield deployment


# @pytest.fixture(scope="module", autouse=True)
# def teardown_module(request, federation, global_config):
#     def finalizer():
#         if global_config['keep_deployment']:
#             log.info("Keeping the deployment as per the request")
#         else:
#             delete_federation(federation)
#             log.info("Federation deleted")

#     request.addfinalizer(finalizer)
