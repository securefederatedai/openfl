#!/bin/bash
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if python3 setup.py sdist bdist_wheel ; then
   echo "Pip wheel built and installed in dist directory"
fi
