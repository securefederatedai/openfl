from socket import getfqdn

class Federation:
    def __init__(self, central_node_fqdn=None, disable_tls=False) -> None:
        if central_node_fqdn is None:
            self.fqdn = getfqdn()
        else:
            self.fqdn = central_node_fqdn

        self.disable_tls = disable_tls


    def register_collaborators(self, col_data_paths: dict) -> None:
        self.col_data_paths = col_data_paths
        with open('./data.yaml', 'w') as f:
            for col_name, data_path in self.col_data_paths.items():
                f.write(f'{col_name},{data_path}\n')