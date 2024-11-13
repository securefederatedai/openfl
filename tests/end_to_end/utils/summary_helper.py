# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from lxml import etree
import os

# Initialize the XML parser
parser = etree.XMLParser(recover=True, encoding='utf-8')
tree = ET.parse("results/results.xml", parser=parser)

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
    if not os.path.exists(agg_log_file):
        print(f"Aggregator log file {agg_log_file} not found. Cannot get aggregated accuracy")
        return "Not Found"

    # Example line(s) containing spaces and special characters:
    """
    METRIC   {'metric_origin': 'aggregator', 'task_name': 'aggregated_model_validation', 'metric_name': 'accuracy', 'metric_value':     aggregator.py:933
        0.15911591053009033, 'round': 0}
    """
    try:
        with open(agg_log_file, 'r') as f:
            for line in f:
                if "metric_origin" in line and "aggregator" in line and "aggregated_model_validation" in line:
                    line = line.split("aggregator.py:")[0].strip()
                    # If the line does not contain closing bracket "}", then concatenate the next line
                    reqd_line = line if "}" in line else line + next(f).strip()
                    agg_accuracy = eval(reqd_line.split("METRIC")[1].strip('"'))["metric_value"]
        return agg_accuracy

    except Exception as e:
        # Do not fail the test if the accuracy cannot be fetched
        print(f"Error while reading aggregator log file: {e}")
        return "Not Found"


def get_test_status(result):
    """
    Get the test status/verdict
    Args
        result: the result object to check`
    Returns
        status of the test status
    """
    status = "FAILED"
    if "failure" in result.tag or "error" in result.tag:
        # If the result has a tag "failure", set status as "FAIL"
        status = "FAILED"
    elif "skipped" in result.tag:
        # If the result has a tag "skipped", set status as "SKIPPED"
        status = "SKIPPED"
    else:
        status = "PASSED"
    return status


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

                # Iterate over each result in testsuite
                for result in testcase:
                    status = get_test_status(result)
                    database_dict["result"] = status

                # Append the dictionary to database_list
                database_list.append(database_dict)
                status = None

    return database_list


if __name__ == "__main__":
    """
    Main function to get the test case results and aggregator logs
    And write the results to GitHub step summary
    """
    result = get_testcase_result()

    num_cols = os.getenv("NUM_COLLABORATORS")
    num_rounds = os.getenv("NUM_ROUNDS")
    model_name = os.getenv("MODEL_NAME")

    if not model_name:
        print("MODEL_NAME is not set, cannot find out aggregator logs")
    else:
        workspace_name = "workspace_" + model_name
        agg_log_file = os.path.join("results", workspace_name, "aggregator.log")
        agg_accuracy = get_aggregated_accuracy(agg_log_file)

    # Write the results to GitHub step summary
    with open(os.getenv('GITHUB_STEP_SUMMARY'), 'a') as fh:
        # DO NOT change the print statements
        print("| Name | Time (in seconds) | Result | Collaborators | Rounds to train | Score (if applicable) |", file=fh)
        print("| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |", file=fh)
        for item in result:
            print(f"| {item['name']} | {item['time']} | {item['result']} | {num_cols} | {num_rounds} | {agg_accuracy} |", file=fh)
