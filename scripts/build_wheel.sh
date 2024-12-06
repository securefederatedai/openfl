#!/bin/bash
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

python3 setup.py sdist bdist_wheel
echo "Pip wheel created under dist/"
