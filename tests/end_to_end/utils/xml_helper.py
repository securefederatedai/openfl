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


def get_aggregator_logs(model_name):
    """
    Get the aggregator logs to fetch the metric values and scores
    Args:
        model_name: the model name for which the aggregator logs are to be fetched
    Returns:
        tuple: the locally tuned model validation, train, aggregated model validation and score
    """
    lt_mv, train, agg_mv, score = None, None, None, "NA"

    workspace_name = "workspace_" + model_name
    agg_log_file = os.path.join("results", workspace_name, "aggregator.log")

    if not os.path.exists(agg_log_file):
        print(f"Aggregator log file {agg_log_file} not found.")
    else:
        with open(agg_log_file, 'r') as f:
            for raw_line in f:
                # Log file contains aggregator.py:<line no> which gets concatenated with the actual log line if not stripped
                line = raw_line.strip() if "aggregator.py:" not in raw_line else raw_line.split("aggregator.py:")[0].strip()
                # Fetch the metric origin and aggregator details
                if "metric_origin" in line and "aggregator" in line:
                    if "locally_tuned_model_validation" in line:
                        reqd_line = line.strip() if "}" in line else line.strip() + next(f).strip()
                        lt_mv = eval(reqd_line.split("METRIC")[1].strip('"'))
                    if "train" in line:
                        reqd_line = line.strip() if "}" in line else line.strip() + next(f).strip()
                        train = eval(reqd_line.split("METRIC")[1].strip('"'))
                    if "aggregated_model_validation" in line:
                        reqd_line = line.strip() if "}" in line else line.strip() + next(f).strip()
                        agg_mv = eval(reqd_line.split("METRIC")[1].strip('"'))

                # Fetch the best model details
                if "saved the best model" in line:
                    reqd_line = line.strip()
                    score_line = reqd_line.split("METRIC")[1].strip('"').strip()
                    score = score_line.split("score")[1].strip()

    return (lt_mv, train, agg_mv, score)


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
    score = "NA"
    result = get_testcase_result()

    if not os.getenv("MODEL_NAME"):
        print("MODEL_NAME is not set, cannot find out aggregator logs")
    else:
        (lt_mv, train, agg_mv, score) = get_aggregator_logs(os.getenv("MODEL_NAME"))

    num_cols = os.getenv("NUM_COLLABORATORS")
    num_rounds = os.getenv("NUM_ROUNDS")
    # Write the results to GitHub step summary
    with open(os.getenv('GITHUB_STEP_SUMMARY'), 'a') as fh:
        # DO NOT change the print statements
        print("| Name | Time (in seconds) | Result | Score (if applicable) | Collaborators | Rounds to train |", file=fh)
        print("| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |", file=fh)
        for item in result:
            print(f"| {item['name']} | {item['time']} | {item['result']} | {score} | {num_cols} | {num_rounds} |", file=fh)
        print("", file=fh)

        # DO NOT change the print statements
        if lt_mv and train and agg_mv:
            print("| Task | Metric Name | Metric Value | Round |", file=fh)
            print("| ------------- | ------------- | ------------- | ------------- |", file=fh)
            print(f"| {lt_mv['task_name']} | {lt_mv['metric_name']} | {lt_mv['metric_value']} | {int(lt_mv['round'] + 1)} |", file=fh)
            print(f"| {train['task_name']} | {train['metric_name']} | {train['metric_value']} | {int(train['round'] + 1)} |", file=fh)
            print(f"| {agg_mv['task_name']} | {agg_mv['metric_name']} | {agg_mv['metric_value']} | {int(agg_mv['round'] + 1)} |", file=fh)
