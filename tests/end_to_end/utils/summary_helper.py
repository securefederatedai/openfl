# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from defusedxml.ElementTree import parse as defused_parse
from lxml import etree
import os
from pathlib import Path

import tests.end_to_end.utils.constants as constants

# Initialize the XML parser
parser = etree.XMLParser(recover=True, encoding="utf-8")

result_path = os.path.join(Path().home(), "results")
result_xml = os.path.join(result_path, "results.xml")
if not os.path.exists(result_xml):
    print(f"Results XML file not found at {result_xml}. Exiting...")
    exit(1)

tree = defused_parse(result_xml, parser=parser)

# Get the root element
testsuites = tree.getroot()


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

    # Example line(s) containing spaces and special characters:
    """
    METRIC   {'metric_origin': 'aggregator', 'task_name': 'aggregated_model_validation', 'metric_name': 'accuracy', 'metric_value':     aggregator.py:933
        0.15911591053009033, 'round': 0}
    """
    try:
        with open(agg_log_file, 'r') as f:
            for line in f:
                if "'metric_origin': 'aggregator'" in line and "aggregated_model_validation" in line:
                    # In Python versions < 3.11, aggregator.py file name appears in the line
                    # whereas in Python version 3.11, it is utils.py
                    line = line.split("aggregator.py:")[0].strip()
                    line = line.split("utils.py:")[0].strip()
                    # If the line does not contain closing bracket "}", then concatenate the next line
                    reqd_line = line if "}" in line else line + next(f).strip()
                    agg_accuracy = eval(reqd_line.split("METRIC")[1].strip('"'))["metric_value"]
    except Exception as e:
        # Do not fail the test if the accuracy cannot be fetched
        print(f"Error while reading aggregator log file: {e}")

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


def main():
    """
    Main function to get the test case results and aggregator logs
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
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")

    # Validate the model name and create the workspace name
    if not model_name.upper() in constants.ModelName._member_names_:
        print(
            f"Invalid model name: {model_name}. Skipping writing to GitHub step summary"
        )
        return

    # Assumption - result directory is present in the home directory
    agg_log_file = os.path.join(
        result_path, model_name, "aggregator", "workspace", "aggregator.log"
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


if __name__ == "__main__":
    main()
