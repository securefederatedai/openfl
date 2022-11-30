from pathlib import Path
import socket
import os
from cryptography import x509
from cryptography.hazmat.backends import default_backend

from openfl.utilities.workspace import set_directory


PREFIX = Path('~/.local/workspace').expanduser()
TEMPLATE = 'keras_cnn_mnist'


def extract_common_name(csr_path):
    with open(csr_path, 'rb') as csr:
        request = csr.read()
        info = x509.load_pem_x509_csr(request, backend=default_backend)
        return info.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value


if __name__ == '__main__':
    PREFIX.mkdir(parents=True, exist_ok=True)
    with set_directory(PREFIX):
        from openfl.interface import workspace
        from openfl.interface import aggregator
        workspace.create(PREFIX, TEMPLATE)
        origin_dir = os.getcwd()

        workspace.certify()
        fqdn = socket.getfqdn()
        aggregator.generate_cert_request(fqdn)
        common_name = extract_common_name(f'cert/server/agg_{fqdn}.csr')
        assert common_name == fqdn
        print('[SUCCESS]')
        print('Aggregator certificate matches the fully qualified domain name')