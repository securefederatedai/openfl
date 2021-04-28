# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Federation API module."""

from socket import getfqdn


class Federation:
    """
    Federation class.

    Federation entity exists to keep information about collaborator related settings,
    their local data and network setting to enable communication in federation.
    """

    def __init__(self, central_node_fqdn=None, disable_tls=False,
                 cert_chain=None, agg_certificate=None, agg_private_key=None) -> None:
        """
        Initialize federation.

        Federation API class should be initialized with the aggregator node FQDN
        and encryption settings. One may disable mTLS in trusted environments or
        provide paths to a certificate chain to CA, aggregator certificate and
        pricate key to enable mTLS.
        """
        if central_node_fqdn is None:
            self.fqdn = getfqdn()
        else:
            self.fqdn = central_node_fqdn

        self.disable_tls = disable_tls

        self.cert_chain = cert_chain
        self.agg_certificate = agg_certificate
        self.agg_private_key = agg_private_key

    def register_collaborators(self, col_data_paths: dict) -> None:
        """
        Provide data to be stored in data.yaml.

        This method should be used to provide information about collaborators
        participating federation.
        Arguments:
        col_data_paths: dict(collaborator name : local data path)
        """
        self.col_data_paths = col_data_paths
        with open('./data.yaml', 'w') as f:
            for col_name, data_path in self.col_data_paths.items():
                f.write(f'{col_name},{data_path}\n')
