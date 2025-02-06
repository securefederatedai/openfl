# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from defusedxml.ElementTree import parse as defused_parse
from lxml import etree
import os
import re
from pathlib import Path

import tests.end_to_end.utils.constants as constants
from tests.end_to_end.utils.db_helper import DBHelper

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


def get_best_agg_score(database_file):
    """
    Get the best_score from the database
    Args:
        database_file: the database file
    Returns:
        best_agg_score: the best score
    """
    best_agg_score = "Not Found"
    if not os.path.exists(database_file):
        print(f"Database file {database_file} not found. Cannot get best aggregated score")
        return best_agg_score

    db_obj = DBHelper(database_file)
    round_number, best_agg_score = db_obj.read_key_value_store()
    print(f"Best aggregated score: {best_agg_score} is in round_number {round_number} ")
    return best_agg_score


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
    model_name = os.getenv("MODEL_NAME").replace("/", "_")
    summary_file = _get_summary_file()

    # Validate the model name and create the workspace name
    if not model_name.upper() in constants.ModelName._member_names_:
        print(
            f"Invalid model name: {model_name}. Skipping writing to GitHub step summary"
        )
        return

    # Assumption - result directory is present in the home directory
    tensor_db_file = os.path.join(
        result_path,
        model_name,
        "aggregator",
        "workspace",
        "local_state",
        "tensor.db",
    )
    best_score = get_best_agg_score(tensor_db_file)

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
                f"| {item['name']} | {item['time']} | {item['result']} | {item['err_msg']} | {num_cols} | {num_rounds} | {best_score} |",
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
