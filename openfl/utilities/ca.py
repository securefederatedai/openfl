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

"""Generic check functions."""
import os


def get_credentials(folder_path):
    """Get credentials from folder by template."""
    root_ca, key, cert = None, None, None
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if ".key" in f:
                key = folder_path + os.sep + f
            if ".crt" in f and "root_ca" not in f:
                cert = folder_path + os.sep + f
            if "root_ca" in f:
                root_ca = folder_path + os.sep + f
    return root_ca, key, cert
