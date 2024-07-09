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

"""Custom input types definition for Click"""

import ast

import click

from openfl.utilities import utils


class FqdnParamType(click.ParamType):
    """Domain Type for click arguments."""

    name = "fqdn"

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it."""
        if not utils.is_fqdn(value):
            self.fail(f"{value} is not a valid domain name", param, ctx)
        return value


class IpAddressParamType(click.ParamType):
    """IpAddress Type for click arguments."""

    name = "IpAddress type"

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it."""
        if not utils.is_api_adress(value):
            self.fail(f"{value} is not a valid ip adress name", param, ctx)
        return value


class InputSpec(click.Option):
    """List or dictionary that corresponds to the input shape for a model"""

    def type_cast_value(self, ctx, value):
        try:
            if value is None:
                return None
            else:
                return ast.literal_eval(value)
        except Exception:
            raise click.BadParameter(value)


FQDN = FqdnParamType()
IP_ADDRESS = IpAddressParamType()
