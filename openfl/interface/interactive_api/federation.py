from socket import getfqdn

class Federation:
    def __init__(self, central_node_fqdn=None, disable_tls=False,
        cert_chain=None, agg_certificate=None, agg_private_key=None) -> None:
        if central_node_fqdn is None:
            self.fqdn = getfqdn()
        else:
            self.fqdn = central_node_fqdn

        self.disable_tls = disable_tls

        self.cert_chain = cert_chain
        self.agg_certificate = agg_certificate
        self.agg_private_key = agg_private_key


    def register_collaborators(self, col_data_paths: dict) -> None:
        self.col_data_paths = col_data_paths
        with open('./data.yaml', 'w') as f:
            for col_name, data_path in self.col_data_paths.items():
                f.write(f'{col_name},{data_path}\n')