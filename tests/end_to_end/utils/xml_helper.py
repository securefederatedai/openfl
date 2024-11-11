# Copyright 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from lxml import etree
import os

from tests.end_to_end.utils.logger import logger as log

# Initialize the XML parser
parser = etree.XMLParser(recover=True, encoding='utf-8')
tree = ET.parse("results/results.xml", parser=parser)

# Get the root element
testsuites = tree.getroot()


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

    log.info(f"Database list = {database_list}")
    return database_list


result = get_testcase_result()

# Write the results to GitHub step summary
with open(os.getenv('GITHUB_STEP_SUMMARY'), 'a') as fh:
    # DO NOT change the print statements
    print("| Name | Time (in seconds) | Result |", file=fh)
    print("| ------------- | ------------- | ------------- |", file=fh)
    for item in result:
        print(f"| {item['name']} | {item['time']} | {item['result']} |", file=fh)
