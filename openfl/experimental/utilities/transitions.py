# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect criteria for transitions in placement."""


def should_transfer(func, parent_func):
    if collaborator_to_aggregator(func, parent_func):
        return True
    else:
        return False


def aggregator_to_collaborator(func, parent_func):
    if parent_func.aggregator_step and func.collaborator_step:
        return True
    else:
        return False


def collaborator_to_aggregator(func, parent_func):
    if parent_func.collaborator_step and func.aggregator_step:
        return True
    else:
        return False
