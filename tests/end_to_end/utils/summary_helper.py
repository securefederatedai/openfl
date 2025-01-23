# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from defusedxml.ElementTree import parse as defused_parse
from lxml import etree
import os
import re
from pathlib import Path

import tests.end_to_end.utils.constants as constants
from tests.end_to_end.utils.generate_report import convert_to_json

result_path = os.path.join(Path().home(), "results")

def initialize_xml_parser():
    """
    Initialize the XML parser and parse the results XML file.
    Returns:
        testsuites: the root element of the parsed XML tree
    """
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    result_xml = os.path.join(result_path, "results.xml")
    if not os.path.exists(result_xml):
        print(f"Results XML file not found at {result_xml}. Exiting...")
        exit(1)

    tree = defused_parse(result_xml, parser=parser)

    # Get the root element
    testsuites = tree.getroot()
    return testsuites


def get_aggregated_accuracy(agg_log_file):
    """
    Get the aggregated accuracy from aggregator logs
    Args:
        agg_log_file: the aggregator log file
    Returns:
        agg_accuracy: the aggregated accuracy
    """
    agg_accuracy = "Not Found"
    if not os.path.exists(agg_log_file):
        print(
            f"Aggregator log file {agg_log_file} not found. Cannot get aggregated accuracy"
        )
        return agg_accuracy

    agg_accuracy_dict = convert_to_json(agg_log_file)

    if not agg_accuracy_dict:
        print(f"Aggregator log file {agg_log_file} is empty. Cannot get aggregated accuracy, returning 'Not Found'")
    else:
        agg_accuracy = agg_accuracy_dict[-1].get(
            "aggregator/aggregated_model_validation/accuracy", "Not Found"
        )
    return agg_accuracy


def get_test_status(result):
    """
    Get the test status/verdict
    Args
        result: the result object to check`
    Returns
        status of the test status
    """
    status, err_msg = "FAILED", "NA"
    if "failure" in result.tag or "error" in result.tag:
        # If the result has a tag "failure", set status as "FAIL"
        status = "FAILED"
        err_msg = result.get("message").split("\n")[0]
    elif "skipped" in result.tag:
        # If the result has a tag "skipped", set status as "SKIPPED"
        status = "SKIPPED"
    else:
        status = "PASSED"
    return status, err_msg


def get_testcase_result():
    """
    Get the test case results from the XML file
    """
    database_list = []
    status = None
    # Initialize the XML parser
    testsuites = initialize_xml_parser()
    # Iterate over each testsuite in testsuites
    for testsuite in testsuites:
        # Populate testcase details in a dictionary
        for testcase in testsuite:
            database_dict = {}
            if testcase.attrib.get("name"):
                database_dict["name"] = testcase.attrib.get("name")
                database_dict["time"] = testcase.attrib.get("time")

                # Successful test won't have any result/subtag
                if len(testcase) == 0:
                    database_dict["result"] = "PASSED"
                    database_dict["err_msg"] = "NA"

                # Iterate over each result in testsuite
                for result in testcase:
                    status, err_msg = get_test_status(result)
                    database_dict["result"] = status
                    database_dict["err_msg"] = err_msg

                # Append the dictionary to database_list
                database_list.append(database_dict)
                status = None

    return database_list


def print_task_runner_score():
    """
    Function to get the test case results and aggregator logs
    And write the results to GitHub step summary
    IMP: Do not fail the test in any scenario
    """
    result = get_testcase_result()

    if not all(
        [
            os.getenv(var)
            for var in [
                "NUM_COLLABORATORS",
                "NUM_ROUNDS",
                "MODEL_NAME",
                "GITHUB_STEP_SUMMARY",
            ]
        ]
    ):
        print(
            "One or more environment variables not set. Skipping writing to GitHub step summary"
        )
        return

    num_cols = os.getenv("NUM_COLLABORATORS")
    num_rounds = os.getenv("NUM_ROUNDS")
    model_name = os.getenv("MODEL_NAME")
    summary_file = _get_summary_file()

    # Validate the model name and create the workspace name
    if not model_name.upper() in constants.ModelName._member_names_:
        print(
            f"Invalid model name: {model_name}. Skipping writing to GitHub step summary"
        )
        return

    # Assumption - result directory is present in the home directory
    agg_log_file = os.path.join(
        result_path,
        model_name,
        "aggregator",
        "workspace",
        "logs",
        "aggregator_metrics.txt",
    )
    agg_accuracy = get_aggregated_accuracy(agg_log_file)

    # Write the results to GitHub step summary file
    # This file is created at runtime by the GitHub action, thus we cannot verify its existence beforehand
    with open(summary_file, "a") as fh:
        # DO NOT change the print statements
        print(
            "| Name | Time (in seconds) | Result | Error (if any) | Collaborators | Rounds to train | Score (if applicable) |",
            file=fh,
        )
        print(
            "| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |",
            file=fh,
        )
        for item in result:
            print(
                f"| {item['name']} | {item['time']} | {item['result']} | {item['err_msg']} | {num_cols} | {num_rounds} | {agg_accuracy} |",
                file=fh,
            )


def print_federated_runtime_score():
    """
    Function to get the federated runtime score from the director log file
    And write the results to GitHub step summary
    IMP: Do not fail the test in any scenario
    """
    summary_file = _get_summary_file()
    search_string = "Aggregated model validation score"

    last_occurrence = aggregated_model_score = None

    # Assumption - result directory is present in the home directory
    dir_res_file = os.path.join(
        result_path,
        "301_mnist_watermarking",
        "director.log",
    )

    # Open and read the log file
    with open(dir_res_file, "r") as file:
        for line in file:
            if search_string in line:
                last_occurrence = line

    # Extract the value from the last occurrence
    if last_occurrence:
        match = re.search(
            r"Aggregated model validation score = (\d+\.\d+)", last_occurrence
        )
        if match:
            aggregated_model_score = match.group(1)
            print(f"Last Aggregated model validation score: {aggregated_model_score}")
        else:
            print("No valid score found in the last occurrence.")
    else:
        print(f"No occurrences of '{search_string}' found in the log file.")

    # Write the results to GitHub step summary file
    # This file is created at runtime by the GitHub action, thus we cannot verify its existence beforehand
    with open(summary_file, "a") as fh:
        # DO NOT change the print statements
        print("| Aggregated model validation score |", file=fh)
        print("| ------------- |", file=fh)
        print(f"| {aggregated_model_score} |", file=fh)


def _get_summary_file():
    """
    Function to get the summary file path
    Returns:
        summary_file: Path to the summary file
    """
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    print(f"Summary file: {summary_file}")

    # Check if the fetched summary file is valid
    if summary_file and os.path.isfile(summary_file):
        return summary_file
    else:
        print("Invalid summary file. Exiting...")
        exit(1)


def fetch_args():
    """
    Function to fetch the commandline arguments.
    Returns:
        Parsed arguments
    """
    # Initialize the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--func_name", required=True, default="", type=str, help="Name of function to be called"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Fetch input arguments
    args = fetch_args()
    func_name = args.func_name
    if func_name in ["print_task_runner_score", "print_local_runtime_score"]:
        print_task_runner_score()
    elif func_name == "print_federated_runtime_score":
        print_federated_runtime_score()
