# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Federation API module."""

from socket import getfqdn
from .shard_descriptor import DummyShardDescriptor
from openfl.transport.grpc.director_client import DirectorClient


class Federation:
    """
    Federation class.

    Federation entity exists to keep information about collaborator related settings,
    their local data and network setting to enable communication in federation.
    """

    def __init__(self, director_node_fqdn=None, disable_tls=False,
                 cert_chain=None, API_cert=None, API_private_key=None) -> None:
        """
        Initialize federation.

        Federation API class should be initialized with the Director node FQDN
        and encryption settings. One may disable mTLS in trusted environments or
        provide paths to a certificate chain to CA, API certificate and
        pricate key to enable mTLS.

        Args:
        - director_node_fqdn: Address and port a director's service is running on.
            User passes here an address with a port.
        """
        if director_node_fqdn is None:
            self.director_node_fqdn = getfqdn()
        else:
            self.director_node_fqdn = director_node_fqdn

        self.disable_tls = disable_tls

        self.cert_chain = cert_chain
        self.API_cert = API_cert
        self.API_private_key = API_private_key

        # Create Director client
        self.dir_client = DirectorClient(director_node_fqdn)

        self.sample_shape, self.target_shape = self._request_data_shape()

    def get_dummy_shard_descriptor(self, size):
        return DummyShardDescriptor(self.sample_shape, self.target_shape, size)

    def get_shard_registry(self):
        return self.dir_client.request_shard_registry()

    def _request_data_shape(self):
        """
        Request sample and target shapes from Director.

        This is an internal method for finding out dataset properties in a Federation.
        """
        sample_shape, target_shape = self.dir_client.get_shard_info()
        return sample_shape, target_shape
