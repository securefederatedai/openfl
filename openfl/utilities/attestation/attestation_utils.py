# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import http
import json
import logging
import os

import requests

from openfl.cryptography.signer import ECDSASigner

logger = logging.getLogger(__name__)


class AttestedIdentity:
    """Class to represent an attested identity for a participant enclave."""

    def __init__(self, mrenclave, name, token, cert, private_key, public_key, root_cert_path):
        """Initializes the AttestedIdentity with the provided parameters.

        Args:
            mrenclave (str): MRENCLAVE value of the enclave.
            name (str): Name of the enclave.
            token (str): AVS report token.
            cert (bytes): Certificate for mTLS communication.
            private_key (bytes): Private key for mTLS communication.
        """
        self.mrenclave = mrenclave
        self.name = name
        self.token = token
        self.cert = cert
        self.private_key = private_key
        self.public_key = public_key
        self.root_cert_path = root_cert_path

    # define getter/seetter methods for the attributes if needed

    def get_mrenclave(self):
        """Returns the MRENCLAVE value of the enclave."""
        return self.mrenclave

    def get_name(self):
        """Returns the name of the enclave."""
        return self.name

    def get_token(self):
        """Returns the AVS report token."""
        return self.token

    def get_cert(self):
        """Returns the certificate for mTLS communication."""
        return self.cert

    def get_private_key(self):
        """Returns the private key for mTLS communication."""
        return self.private_key

    def set_mrenclave(self, mrenclave):
        """Sets the MRENCLAVE value of the enclave."""
        self.mrenclave = mrenclave

    def get_passport(self):
        """Returns a dictionary representation of the attested identity.

        Returns:
            dict: A dictionary containing the MRENCLAVE value, participant name,
                  AVS report token, certificate, and private key.
        """
        return {
            "mrenclave": self.mrenclave,
            "name": self.name,
            "token": self.token,
            "cert": self.cert,
        }

    def save_passport(self, path):
        """Saves the attested identity passport to a file.

        Args:
            path (str): Path to save the passport file.
        """
        if not os.path.exists(os.path.dirname(path)):
            # give an error if the directory does not exist
            raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist")

        passport = self.get_passport()
        with open(path, "w") as f:
            json.dump(passport, f, indent=4)

    def save_token(self, path):
        """Saves the AVS report token to a file.

        Args:
            path (str): Path to save the token file.
        """
        if not os.path.exists(os.path.dirname(path)):
            # give an error if the directory does not exist
            raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist")

        with open(path, "w") as f:
            f.write(self.token)

    def save_pubkey(self, path):
        """Saves the public key to a file.

        Args:
            path (str): Path to save the public key file.
        """
        if not os.path.exists(os.path.dirname(path)):
            # give an error if the directory does not exist
            raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist")

        with open(path, "wb") as f:
            f.write(self.public_key)

    def save_cert(self, path):
        """Saves the certificate to a file.

        Args:
            path (str): Path to save the certificate file.
        """
        if not os.path.exists(os.path.dirname(path)):
            # give an error if the directory does not exist
            raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist")

        with open(path, "wb") as f:
            f.write(self.cert)

    def save_private_key(self, path):
        """Saves the private key to a file.

        Args:
            path (str): Path to save the private key file.
        """
        if not os.path.exists(os.path.dirname(path)):
            # give an error if the directory does not exist
            raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist")

        with open(path, "wb") as f:
            f.write(self.private_key)

    def build_root_cert(self, client_certs_path):
        """Builds the root certificate for the attested identity.

        Returns:
            str: Path to the root certificate.
        """
        if not os.path.exists(client_certs_path):
            raise FileNotFoundError(f"Client certs path {client_certs_path} does not exist")

        agg_cert = self.get_cert()
        # find all .crt files in client cert path and add them to the root cert
        client_certs = []
        for root, dirs, files in os.walk(client_certs_path):
            for file in files:
                if file.endswith(".crt"):
                    logger.info(f"Found client cert: {file} in {root}")
                    client_certs.append(os.path.join(root, file))

        # create a root cert with the agg cert and client certs
        root_cert_chain = os.path.join(
            os.path.dirname(self.get_root_cert_path()),
            "cert_chain.crt",
        )
        with open(root_cert_chain, "wb") as f:
            f.write(agg_cert)
            for client_cert in client_certs:
                with open(client_cert, "rb") as cf:
                    f.write(cf.read())
        logger.info(f"Root cert created at {root_cert_chain}")
        # print the root cert
        with open(root_cert_chain, "rb") as f:
            root_cert = f.read()
            logger.info(f"Root cert: {root_cert}")

        return self.root_cert_path

    def get_root_cert_path(self):
        """Returns the path to the attestation report.

        Returns:
            str: Path to the attestation report.
        """
        return self.root_cert_path


class AttestationManager:
    """Class to manage attestation for participant enclaves.
    This class handles the generation of SGX quotes, fetching MRENCLAVE values,
    and obtaining attestation reports from AVS (Attestation Verification Service).
    """

    def __init__(
        self,
        participant_name,
        attestation_report_path,
        ita_api_key,
        avs_url,
        root_cert_path,
        privkey_path=None,
    ):
        """Initializes the AttestationManager with the provided parameters.

        Args:
            participant_name (str): Name of the enclave.
            attestation_report_path (str): Path to the store attestation reports.
            ita_api_key (str): API key for ITA.
            avs_url (str): URL for AVS.
        """
        self.participant_name = participant_name
        self.attestation_report_path = attestation_report_path
        self.ita_api_key = ita_api_key
        self.avs_url = avs_url
        self.root_cert_path = root_cert_path
        logger.info(f"root cert path: {self.root_cert_path}")
        self.privkey_path = privkey_path
        if self.privkey_path is None:
            self.ecdsa_p_384_signer = ECDSASigner.get_instance()
        else:
            self.ecdsa_p_384_signer = ECDSASigner.get_instance(self.privkey_path)

    def get_attested_identity(self, cert_host="localhost"):
        """Generates an attested identity, where a public/private key pair
        is bound to the workload measurement (MRENCLAVE) via
        a remote attestation report.
        Args:
            None
        Raises:
            FileNotFoundError: If the attestation report path does not exist.
            ValueError: If the ITA API key or AVS URL is not set.
        Returns:
            dict: A dictionary containing the MRENCLAVE value, participant name,
                  and AVS report token.
        """

        if not os.path.exists(self.attestation_report_path):
            raise FileNotFoundError(
                f"attestation report path {self.attestation_report_path} does not exist"
            )

        if self.ita_api_key is None:
            raise ValueError("ITA API key is required for remote attestation")
        if self.avs_url is None:
            raise ValueError("AVS URL is required for remote attestation")

        mrenclave = self.fetch_mrenclave_from_quote(None)
        logger.info(f"Enclave MRENCLAVE: {mrenclave}")

        # Generate ECDSA-P 384 key pair and certificate
        # This is the enclave specific ECDSA-P 384 keypair for setting up mTLS
        # with collaborator enclave

        os.path.join(self.attestation_report_path, f"{self.participant_name}_privkey.pem")
        cert = self.ecdsa_p_384_signer.cert(cert_host, mrenclave).encode("utf-8")
        key = self.ecdsa_p_384_signer.serialize_private_key()
        self_signed_cert_path = os.path.join(
            self.attestation_report_path, f"{self.participant_name}_pubkey.pem"
        )

        self.pubkey = self.ecdsa_p_384_signer.get_pubkey()
        avs_report = self.get_avs_report_ita(
            self.participant_name,
            self.attestation_report_path,
            cert,
            self.avs_url,
            self.ita_api_key,
        )
        logger.info(f"AVS report: {avs_report}")
        attested_identity = AttestedIdentity(
            mrenclave=mrenclave,
            name=self.participant_name,
            token=avs_report["token"],
            cert=cert,
            private_key=key,
            public_key=self.pubkey,
            root_cert_path=self.root_cert_path,
        )
        # save public key to file
        attested_identity.save_cert(self_signed_cert_path)

        return attested_identity

    def get_ecdsa_p_384_signer(self):
        """Returns the ECDSA-P 384 signer instance.

        Returns:
            ECDSASigner: The ECDSA-P 384 signer instance.
        """
        return self.ecdsa_p_384_signer

    def gen_sgx_quote(
        self,
        user_data_bytes=None,
        quote_dump_path="/tmp/quote.json",
        orig_data=None,
        challenge="0000000000000",
    ):
        """Generates the SGX quote and saves it to the specified path.
        Args:
            user_data_bytes (bytes): User data to be included in the quote.
            quote_dump_path (str): Path to save the generated quote.
            orig_data (bytes): Original data to be included in the quote.
            challenge (str): Challenge string to be included in the quote.
        Returns:
            dict: A dictionary containing the generated quote.
        Raises:
            ValueError: If the user data is not a bytes object or exceeds 64 bytes.
        """

        # Set user data
        if user_data_bytes is not None:
            # Check that user data is a bytes like object
            if isinstance(user_data_bytes, bytes) is not True:
                raise ValueError("User data must be a bytes object")

            # Check that user data is at most 64 bytes
            if len(list(user_data_bytes)) > 64:
                raise ValueError("User data can be at most 64 bytes")

            # Set user report data
            with open("/dev/attestation/user_report_data", "wb") as fh:
                fh.write(user_data_bytes)

        # Generate attestation quote
        quote = None
        with open("/dev/attestation/quote", "rb") as fh:
            quote = fh.read(8192)

        # Create quote as expected by AVS for verification
        # a. base64 encode the quote
        # b. Set 'userData' to empty, we do not want AVS to verify user data
        quote_avs = {}
        quote_avs["quote"] = base64.b64encode(quote).decode("utf-8")

        if orig_data is not None:
            quote_avs["runtime_data"] = base64.b64encode(orig_data).decode("utf-8")
        else:
            quote_avs["runtime_data"] = ""

        # Save the quote in the specified location
        with open(quote_dump_path, "w") as fh:
            json.dump(quote_avs, fh)

        return quote_avs

    def fetch_mrenclave_from_quote(self, quote=None):
        """Fetches the MRENCLAVE value from the SGX quote.

        Args:
            quote (str): The SGX quote in JSON format.

        Returns:
            str: The MRENCLAVE value extracted from the quote.
        """

        if quote is None:
            # Generate attestation quote
            with open("/dev/attestation/quote", "rb") as fh:
                quote = fh.read(8192)

        # Create quote as expected by AVS for verification
        # a. base64 encode the quote
        # b. Set 'userData' to empty, we do not want AVS to verify user data
        quote_avs = {}
        quote_avs["quote"] = base64.b64encode(quote).decode("utf-8")

        # Decode the base64 encoded quote
        decoded_quote = base64.b64decode(quote_avs["quote"])

        # Extract the MRENCLAVE value from the decoded quote
        mrenclave_hex = decoded_quote[112:144].hex()
        logger.info(f"Extracted MRENCLAVE: {mrenclave_hex}")

        return mrenclave_hex

    def get_avs_report_ita(self, name, attestation_report_path, cert, avs_url, ita_api_key):
        """Fetches the attestation report from AVS using the ITA API key.
        This function generates the SGX quote, sends it to AVS for attestation,
        and saves the attestation report in the specified path.
        Args:
            name (str): Name of the enclave.
            attestation_report_path (str): Path to save the attestation report.
            cert (bytes): Certificate for mTLS with the collaborator enclave.
            avs_url (str): URL for AVS.
            ita_api_key (str): API key for ITA.
        Returns:
            dict: A dictionary containing the attestation report from AVS.
        Raises:
            Exception: If the connection to AVS fails or if the response is not OK.
        """
        # Set the paths for getting the enclave quote and ITA AVS response
        quote_dir = os.path.normpath("/tmp")
        quote_path = os.path.join(quote_dir, f"{name}_quote.json")
        avs_report_path = os.path.join(attestation_report_path, f"{name}_avs_report.json")

        # AVS doesn't work with SHA384, so, moving to SHA256 in the meantime
        cert_sha256_digest = hashlib.sha256(cert).digest()
        self.gen_sgx_quote(cert_sha256_digest, quote_path, cert)

        # ITA attestation
        avs_attest_endpoint = f"{avs_url}/appraisal/v1/attest"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": ita_api_key,
        }

        # Read the quote and send it to AVS for getting the report
        with open(quote_path) as fh:
            quote_avs = fh.read()

        # AVS has a self-signed certificate, so, disabling verification
        res = requests.post(avs_attest_endpoint, headers=headers, data=quote_avs)
        if res.status_code != http.HTTPStatus.OK:
            raise Exception(f"Failed to connect with {avs_url}, err: {res.status_code}")

        avs_report = res.content
        with open(avs_report_path, "wb") as fh:
            fh.write(avs_report)

        # Register enclave with governor
        avs_report = avs_report.decode("utf-8")
        avs_report = json.loads(avs_report)

        return avs_report


def fetch_attestation_env_vars():
    """Fetches attestation environment variables from the system.
    This function retrieves the ITA API key, AVS URL, and attestation report path
    from the environment variables. If the attestation report path is not set,
    it defaults to None.
    Args:
        None
    Raises:
        None

    Returns:
        dict: A dictionary containing the attestation environment variables.
    """
    env_vars = {
        "ITA_API_KEY": os.getenv("ITA_API_KEY"),
        "AVS_URL": os.getenv("AVS_URL"),
        "ATTESTATION_REPORT_PATH": os.getenv("ATTESTATION_REPORT_PATH", None),
        "ROOT_CERT_PATH": os.getenv("ROOT_CERT_PATH", None),
    }
    return env_vars


def get_remote_attestation(participant_name):
    """Starts the remote attestation process for the participant enclave.
    This function initializes the AttestationManager and generates an attested
    identity for the participant enclave.
    Args:
        plan_config (str): Path to the plan configuration file.
        participant_name (str): Name of the enclave.

    Returns:
        AttestedIdentity: An instance of the AttestedIdentity class
        containing the attested identity.
    """
    # Fetch remote attestation environment variables
    attestation_env = fetch_attestation_env_vars()
    attested_identity = None
    if attestation_env is not None:
        attestation_manager = AttestationManager(
            participant_name,
            attestation_env["ATTESTATION_REPORT_PATH"],
            attestation_env["ITA_API_KEY"],
            attestation_env["AVS_URL"],
            attestation_env["ROOT_CERT_PATH"],
        )
        # Generate and store the attestation report
        attested_identity = attestation_manager.get_attested_identity()
        logger.info("Remote attestation report stored successfully.")
    else:
        logger.error("Remote attestation environment variables not set.")
    return attested_identity
