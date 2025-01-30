# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Federation API module."""

from openfl.interface.interactive_api.shard_descriptor import DummyShardDescriptor
from openfl.transport.grpc.director_client import DirectorClient
from openfl.utilities.utils import getfqdn_env


class Federation:
    """Federation class.

    Manages information about collaborator settings, local data, and network settings.

    The Federation class is used to maintain information about collaborator-related settings,
    their local data, and network settings to enable communication within the federation.

    Attributes:
        director_node_fqdn (str): The fully qualified domain name (FQDN) of the director node.
        tls (bool): A boolean indicating whether mTLS (mutual Transport Layer Security) is enabled.
        cert_chain (str): The path to the certificate chain to the Certificate Authority (CA).
        api_cert (str): The path to the API certificate.
        api_private_key (str): The path to the API private key.
        dir_client (DirectorClient): An instance of the DirectorClient class.
        sample_shape (tuple): The shape of the samples in the dataset.
        target_shape (tuple): The shape of the targets in the dataset.
    """

    def __init__(
        self,
        client_id=None,
        director_node_fqdn=None,
        director_port=None,
        tls=True,
        cert_chain=None,
        api_cert=None,
        api_private_key=None,
    ) -> None:
        """
        Initialize federation.

        Federation API class should be initialized with the Director node FQDN
        and encryption settings. One may disable mTLS in trusted environments or
        provide paths to a certificate chain to CA, API certificate and
        pricate key to enable mTLS.

        Args:
            client_id (str): Name of created Frontend API instance.
                The same name user certify.
            director_node_fqdn (str): Address and port a director's service is running on.
                User passes here an address with a port.
            director_port (int): Port a director's service is running on.
            tls (bool): Enable mTLS.
            cert_chain (str): Path to a certificate chain to CA.
            api_cert (str): Path to API certificate.
            api_private_key (str): Path to API private key.
        """
        if director_node_fqdn is None:
            self.director_node_fqdn = getfqdn_env()
        else:
            self.director_node_fqdn = director_node_fqdn

        self.tls = tls

        self.cert_chain = cert_chain
        self.api_cert = api_cert
        self.api_private_key = api_private_key

        # Create Director client
        self.dir_client = DirectorClient(
            client_id=client_id,
            director_host=director_node_fqdn,
            director_port=director_port,
            tls=tls,
            root_certificate=cert_chain,
            private_key=api_private_key,
            certificate=api_cert,
        )

        # Request sample and target shapes from Director.
        # This is an internal method for finding out dataset properties in a Federation.
        self.sample_shape, self.target_shape = self.dir_client.get_dataset_info()

    def get_dummy_shard_descriptor(self, size):
        """Return a dummy shard descriptor.

        Args:
            size (int): Size of the shard descriptor.

        Returns:
            DummyShardDescriptor: A dummy shard descriptor.
        """
        return DummyShardDescriptor(self.sample_shape, self.target_shape, size)

    def get_shard_registry(self):
        """Return a shard registry.

        Returns:
            list: A list of envoys.
        """
        return self.dir_client.get_envoys()
