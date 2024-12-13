# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import logging

from tests.end_to_end.utils.common_fixtures import fx_local_federated_workflow
from tests.end_to_end.workflow.exclude_flow import TestFlowExclude
from tests.end_to_end.workflow.include_exclude_flow import TestFlowIncludeExclude
from tests.end_to_end.workflow.include_flow import TestFlowInclude

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
