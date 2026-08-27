import base64
import os
from datetime import datetime, timedelta

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509.oid import NameOID


class ECDSASigner:
    """ECDSA secp384R1 signer

    This class implements ECDSA methods specific to a single instance of
    enclave, so, that it can be reused across without passing references
    across the breadth of the code. The relevant keys and certificates are
    stored in /tmp. /tmp should be appropriately configured in the enclave
    manifest, so, that it is wiped off when the enclave exits.

    Raises:
        Exception: ValueError() for incorrect configuration

    Returns:
        object: The reference of the only object
    """

    __signer_instance = None

    @staticmethod
    def get_instance(privkey_path="/tmp/client_privkey.pem"):
        if ECDSASigner.__signer_instance is None:
            ECDSASigner(privkey_path)
        return ECDSASigner.__signer_instance

    def __init__(self, privkey_path="/tmp/client_privkey.pem", cert_path="/tmp/"):
        """Constructor for creating the ECDSA Certificate Chain

        Args:
            privkey_path (string, optional): Path to the existing client private key.
                    Defaults to None.

        Raises:
            Exception: Generic, in case there's invalid configuration or file data
        """
        if ECDSASigner.__signer_instance is not None:
            raise Exception("ECDSASigner: Only one instance allowed")
        else:
            ECDSASigner.__signer_instance = self

        self._root_cert_path = os.path.join(cert_path, "openfl-security-ca-cert.pem")
        self._client_cert_path = os.path.join(cert_path, "openfl-enclave-cert.pem")

        # If the private key already exists, then reuse to create the certificate
        # if doesn't exist.
        self._client_cert = None
        self._root_cert = None

        if privkey_path and os.path.exists(privkey_path):
            with open(privkey_path, "rb") as client_priv_fh:
                client_privkey_pem = client_priv_fh.read()

                # Once the private key is found in filesystem, then look for saved
                # certificate and the corresponding public key
                self._client_privkey = load_pem_private_key(client_privkey_pem, password=None)
                if not isinstance(self._client_privkey, ec.EllipticCurvePrivateKey):
                    raise ValueError(f"Invalid private key format: '{privkey_path}'")

                self._client_pubkey = self._client_privkey.public_key()

                # Check if certificate is already present in the filesystem, if not then
                # serialize is not called yet
                if os.path.exists(self._client_cert_path):
                    with open(self._client_cert_path, "rb") as cert_fh:
                        self._client_cert = x509.load_pem_x509_certificate(
                            cert_fh.read(), default_backend()
                        )

                # FIXME: Post upgrading cryptography module, delete this below
                # and change above call to load_pem_x509_certificate(s) to load the chain
                if not os.path.exists(self._root_cert_path):
                    raise ValueError(
                        "Out of tree modification detected, "
                        "clean all the keys and certs and try again"
                    )
                with open(self._root_cert_path, "rb") as cert_fh:
                    self._root_cert = x509.load_pem_x509_certificate(
                        cert_fh.read(), default_backend()
                    )

        else:
            self._root_privkey = ec.generate_private_key(ec.SECP384R1(), default_backend())
            self._root_pubkey = self._root_privkey.public_key()

            self._client_privkey = ec.generate_private_key(ec.SECP384R1(), default_backend())

        self._client_pubkey = self._client_privkey.public_key()

    def __get_cert(
        self,
        subject_name,
        subject_pubkey,
        issuer_name,
        issuer_privkey,
        ca=False,
        mrenclave_data=None,
    ):
        """Create a certificate with optional MRENCLAVE_OID extension.

        Args:
            subject_name (string): The subject name in the certificate, must match the URL.
            subject_pubkey (string): Subject's public key to be embedded in the certificate.
            issuer_name (string): The CA name.
            issuer_privkey (string): The CA private key to sign the subject certificate.
            ca (bool, optional): To set CA=true property. Defaults to False.
            mrenclave_data (string, optional): The mrenclave data to be added as a custom extension.

        Returns:
            object: Certificate.
        """
        # Create the certificate builder
        cert_builder = x509.CertificateBuilder()
        cert_builder = cert_builder.subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_name)])
        )
        cert_builder = cert_builder.issuer_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)])
        )

        oneday = timedelta(1, 0, 0)
        start_validity = datetime.today() - oneday
        end_validity = datetime.today() + (100 * oneday)
        cert_builder = cert_builder.not_valid_before(start_validity)
        cert_builder = cert_builder.not_valid_after(end_validity)

        if ca:
            cert_builder = cert_builder.add_extension(
                x509.BasicConstraints(ca=True, path_length=None), critical=True
            )

        cert_builder = cert_builder.serial_number(x509.random_serial_number())
        cert_builder = cert_builder.public_key(subject_pubkey)

        # Add the custom MRENCLAVE_OID extension if mrenclave_data is provided
        if mrenclave_data:
            MRENCLAVE_OID = x509.ObjectIdentifier("1.3.6.1.4.1.99999.1.1")  # Example OID
            cert_builder = cert_builder.add_extension(
                x509.UnrecognizedExtension(MRENCLAVE_OID, mrenclave_data.encode("utf-8")),
                critical=False,
            )

        # Sign the certificate
        return cert_builder.sign(issuer_privkey, algorithm=hashes.SHA384())

    def __create_cert_chain(self):
        root_cert_bytes = self._root_cert.public_bytes((serialization.Encoding.PEM))
        client_cert_bytes = self._client_cert.public_bytes((serialization.Encoding.PEM))

        # Concatenate together and return
        cert_chain = client_cert_bytes.decode("utf-8") + root_cert_bytes.decode("utf-8")
        return cert_chain

    def cert(self, common_name, mrenclave_data=None):
        """Returns the self-signed certificate
        Args:
            common_name (string): to be used as subject and issuer name
        Returns:
            _type_: _description_
        """

        # Create the certificate chaining with local CA
        # A. Create a root CA certificate
        # B. Create the node certificate

        # Return the cached value if it exists
        if self._client_cert:
            return self.__create_cert_chain()

        ca_common_name = f"{common_name}-CA"
        self._root_cert = self.__get_cert(
            ca_common_name, self._root_pubkey, ca_common_name, self._root_privkey, ca=True
        )

        self._client_cert = self.__get_cert(
            common_name,
            self._client_pubkey,
            ca_common_name,
            self._root_privkey,
            False,
            mrenclave_data=mrenclave_data,
        )

        return self.__create_cert_chain()

    def get_pubkey(self):
        """returns public key as a PEM string"""

        return self._client_pubkey.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    def sign(self, message):
        """sign message string using private key.

        Return Value: base64 encoded string
        """

        signature_bytes = self._client_privkey.sign(
            message.encode("utf-8"), ec.ECDSA(hashes.SHA384())
        )
        return base64.b64encode(signature_bytes).decode("utf-8")

    def serialize_private_key(self, filename="/tmp/client_privkey.pem", save_root_cert=True):
        """write the private key to a file in PEM format"""

        client_privkey_pem = self._client_privkey.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        with open(filename, "wb") as fh:
            fh.write(client_privkey_pem)

        # Save the CA certificate if not already exists
        if os.path.exists(self._root_cert_path) is False and save_root_cert:
            root_cert_bytes = self._root_cert.public_bytes((serialization.Encoding.PEM))
            with open(self._root_cert_path, "wb") as fh:
                fh.write(root_cert_bytes)

        # Save the client certificate if not already exists
        if os.path.exists(self._client_cert_path) is False:
            cert_chain = self.__create_cert_chain()
            with open(self._client_cert_path, "wb") as fh:
                fh.write(cert_chain.encode("utf-8"))

        return client_privkey_pem
