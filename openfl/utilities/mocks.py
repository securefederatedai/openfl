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

"""Mock objects to eliminate extraneous dependencies"""


class MockDataLoader:
    """Placeholder dataloader for when data is not available"""
    def __init__(self, feature_shape):
        self.feature_shape = feature_shape

    def get_feature_shape(self):
        return self.feature_shape
