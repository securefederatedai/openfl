# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Click types module."""

import ipaddress
import re


import click


class DomainParamType(click.ParamType):
    """Domain Type for click arguments."""

    name = 'DomainType'

    @staticmethod
    def is_fqdn(hostname: str) -> bool:
        """https://en.m.wikipedia.org/wiki/Fully_qualified_domain_name."""
        if not 1 < len(hostname) < 253:
            return False

        # Remove trailing dot
        if hostname[-1] == '.':
            hostname = hostname[:-1]

        #  Split hostname into list of DNS labels
        labels = hostname.split('.')

        #  Define pattern of DNS label
        #  Can begin and end with a number or letter only
        #  Can contain hyphens, a-z, A-Z, 0-9
        #  1 - 63 chars allowed
        fqdn = re.compile(r'^[a-z0-9]([a-z-0-9-]{0,61}[a-z0-9])?$', re.IGNORECASE) # noqa 

        # Check that all labels match that pattern.
        return all(fqdn.match(label) for label in labels)

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it."""
        if not self.is_fqdn(value):
            self.fail(f'{value} is not a valid domain name', param, ctx)
        return value


class IpAddressParamType(click.ParamType):
    """IpAddress Type for click arguments."""

    name = 'IpAddress type'

    @staticmethod
    def is_api_adress(address: str) -> bool:
        """Validate ip-adress value."""
        try:
            _ = ipaddress.ip_address(address)
            return True
        except ValueError:
            return False

    def convert(self, value, param, ctx):
        """Validate value, if value is valid, return it."""
        if not self.is_api_adress(value):
            self.fail(f'{value} is not a valid ip adress name', param, ctx)
        return value


DomainType = DomainParamType()
IpAdressType = IpAddressParamType()
