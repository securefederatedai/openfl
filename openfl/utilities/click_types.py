# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Click types module."""

import click
import ast

from openfl.utilities import utils


class FqdnParamType(click.ParamType):
    """Domain Type for click arguments."""

    name = 'fqdn'

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it."""
        if not utils.is_fqdn(value):
            self.fail(f'{value} is not a valid domain name', param, ctx)
        return value


class IpAddressParamType(click.ParamType):
    """IpAddress Type for click arguments."""

    name = 'IpAddress type'

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it."""
        if not utils.is_api_adress(value):
            self.fail(f'{value} is not a valid ip adress name', param, ctx)
        return value

class ListOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            if value is None:
                return None
            else:
                return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


FQDN = FqdnParamType()
IP_ADDRESS = IpAddressParamType()
