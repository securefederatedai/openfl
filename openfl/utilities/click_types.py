# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Custom input types definition for Click"""

import ast

import click

from openfl.utilities import utils


class FqdnParamType(click.ParamType):
    """Domain Type for click arguments.

    This class is used to validate that a command line argument is a fully
    qualified domain name.

    Attributes:
        name (str): The name of the parameter type.
    """

    name = "fqdn"

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it.

        Args:
            value (str): The value to validate.
            param (click.core.Option): The option that this value was supplied
                to.
            ctx (click.core.Context): The context for the parameter.

        Returns:
            str: The value, if it is valid.

        Raises:
            value (click.exceptions.BadParameter): If the value is not a valid
                domain name.
        """
        if not utils.is_fqdn(value):
            self.fail(f"{value} is not a valid domain name", param, ctx)
        return value


class IpAddressParamType(click.ParamType):
    """IpAddress Type for click arguments.

    This class is used to validate that a command line argument is an IP
    address.

    Attributes:
        name (str): The name of the parameter type.
    """

    name = "IpAddress type"

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it.

        Args:
            value (str): The value to validate.
            param (click.core.Option): The option that this value was supplied
                to.
            ctx (click.core.Context): The context for the parameter.

        Returns:
            str: The value, if it is valid.

        Raises:
            click.exceptions.BadParameter: If the value is not a valid IP
                address.
        """
        if not utils.is_api_adress(value):
            self.fail(f"{value} is not a valid ip address name", param, ctx)
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
