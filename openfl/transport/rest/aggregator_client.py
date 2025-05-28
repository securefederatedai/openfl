# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AggregatorRESTClient module."""

# Standard library imports
import logging
import ssl
import struct
import time
from typing import Any, List, Tuple

# Third-party libraries
import requests
from google.protobuf import json_format
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Internal modules
from openfl.protocols import aggregator_pb2, base_pb2
from openfl.protocols.aggregator_client_interface import AggregatorClientInterface

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related error."""

    pass


class AggregatorRESTClient(AggregatorClientInterface):
    def __init__(
        self,
        agg_addr,
        agg_port,
        aggregator_uuid: str,
        federation_uuid: str,
        collaborator_name: str,
        use_tls=True,
        require_client_auth=True,
        root_certificate=None,
        certificate=None,
        private_key=None,
        single_col_cert_common_name=None,
        refetch_server_cert_callback=None,
        **kwargs,
    ):
        """
        Initialize the AggregatorRESTClient with proper security settings.

        Args:
            agg_addr: Aggregator address
            agg_port: Aggregator port
            aggregator_uuid: UUID of the aggregator
            federation_uuid: UUID of the federation
            collaborator_name: Name of the collaborator
            use_tls: Whether to use TLS
            require_client_auth: Whether to require client authentication
            root_certificate: Path to root certificate
            certificate: Path to client certificate
            private_key: Path to client private key
            single_col_cert_common_name: Common name for single collaborator certificate
            refetch_server_cert_callback: Callback to refetch server certificate
        """
        self.use_tls = use_tls
        self.require_client_auth = require_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.collaborator_name = collaborator_name
        self.single_col_cert_common_name = single_col_cert_common_name
        self.refetch_server_cert_callback = refetch_server_cert_callback

        # Determine scheme and TLS verification
        scheme = "https" if self.use_tls else "http"

        # Configure certificate verification
        self.cert_verification = self._configure_cert_verification(
            self.use_tls, self.root_certificate
        )

        # Configure client certificates if required
        if self.use_tls and self.require_client_auth:
            if not self.certificate or not self.private_key:
                raise ValueError(
                    "Both certificate and private key are required for mTLS "
                    "(client authentication). "
                    "Please provide both certificate and private key paths."
                )
            self.cert = (self.certificate, self.private_key)
        else:
            self.cert = None

        # Configure session with proper settings
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update(
            {
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=300",
                "Accept": "application/json",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
            }
        )

        # Configure timeouts with longer duration for large payloads
        self.timeout = (30, 300)  # (connect timeout, read timeout) in seconds

        # Configure retries with backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=True,
        )

        # Configure the adapter with the retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=10, pool_block=False
        )

        # Mount the adapter for both HTTP and HTTPS
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Build the base URL
        self.base_url = f"{scheme}://{agg_addr}:{agg_port}/experimental/v1"

        # Log warning about experimental API
        logger.warning(
            "Initializing Aggregator REST Client (EXPERIMENTAL API - Not for production use)"
        )

        # Verify certificates if TLS is enabled
        if self.use_tls:
            try:
                self._verify_certificates()
            except Exception as e:
                logger.error(f"Certificate verification failed: {e}")
                raise

    @classmethod
    def _configure_cert_verification(
        cls, use_tls: bool, root_certificate: str = None
    ) -> bool | str:
        """
        Configure certificate verification settings for requests.

        Args:
            use_tls: Whether TLS is enabled
            root_certificate: Optional path to root certificate file

        Returns:
            Union[bool, str]: Either True for system CA bundle, False for no verification,
                             or path to root certificate file
        """
        if not use_tls:
            return False

        if root_certificate:
            return root_certificate

        return True  # Use system's default CA bundle

    def _verify_certificates(self):
        """Verify SSL certificates and configuration."""
        import socket
        import ssl

        # Try to establish a test connection
        try:
            hostname = self.base_url.split("://")[1].split(":")[0]
            port = int(self.base_url.split(":")[2].split("/")[0])

            # Create SSL context with specific options
            context = ssl.create_default_context()
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True

            # Set secure cipher suites
            context.set_ciphers("ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384")

            # Disable older TLS versions
            context.options |= (
                ssl.OP_NO_TLSv1
                | ssl.OP_NO_TLSv1_1
                | ssl.OP_NO_TLSv1_2
                | ssl.OP_NO_COMPRESSION
                | ssl.OP_NO_TICKET
            )

            if self.root_certificate:
                context.load_verify_locations(cafile=self.root_certificate)

            if self.certificate and self.private_key:
                context.load_cert_chain(certfile=self.certificate, keyfile=self.private_key)

            # Use context managers for proper resource cleanup
            with socket.create_connection((hostname, port)) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as _:
                    pass  # Connection successful if we get here

        except ssl.SSLError as e:
            if "CERTIFICATE_UNKNOWN" in str(e):
                logger.error(
                    "Certificate unknown error - this usually means the "
                    "server's certificate is not trusted"
                )
                logger.error("Please verify that:")
                logger.error(
                    "1. The root certificate contains all necessary intermediate certificates"
                )
                logger.error("2. The server's certificate is properly signed by a trusted CA")
                logger.error("3. The hostname matches the certificate's subject")
            raise
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise

    def _build_header(self) -> dict:
        """Build and return a header dictionary with security headers."""
        headers = {
            "Receiver": self.aggregator_uuid,
            "Federation-UUID": self.federation_uuid,
            "Single-Col-Cert-CN": self.single_col_cert_common_name or "",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Sender": self.collaborator_name,
        }
        if self.use_tls:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return headers

    def _make_request(
        self, method, url, data=None, params=None, headers=None, stream=False, timeout=None
    ):
        """Make a request with proper security settings."""
        start_time = time.time()
        try:
            self._validate_url_scheme(url)
            request_headers = self._prepare_headers(headers)
            response = self._execute_request(
                method, url, request_headers, data, params, stream, timeout
            )
            self._validate_response(response)
            logger.debug(f"Request completed in {time.time() - start_time:.2f} seconds")
            return response

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {time.time() - start_time:.2f} seconds")
            raise
        except requests.exceptions.ConnectionError as e:
            self._handle_connection_error(e)
            raise
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
            raise

    def _validate_url_scheme(self, url):
        """Validate URL scheme matches TLS setting."""
        if self.use_tls and not url.startswith("https://"):
            raise ValueError("TLS required but URL is not HTTPS")
        elif not url.startswith("http://") and not url.startswith("https://"):
            raise ValueError("URL must use either HTTP or HTTPS scheme")

    def _prepare_headers(self, headers):
        """Prepare request headers with security settings."""
        request_headers = self._build_header()
        if headers:
            request_headers.update(headers)
        return request_headers

    def _execute_request(self, method, url, headers, data, params, stream, timeout):
        """Execute the HTTP request with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                session = requests.Session()
                if self.use_tls:
                    # Extract hostname from URL for verification
                    hostname = url.split("://")[1].split(":")[0].split("/")[0]

                    # Create a custom SSL context for this request
                    context = ssl.create_default_context(
                        cafile=self.root_certificate if self.root_certificate else None
                    )
                    context.verify_mode = ssl.CERT_REQUIRED

                    # Configure session with SSL context and hostname verification
                    session.verify = self.cert_verification
                    session.cert = self.cert

                    # Configure adapter with proper SSL settings
                    adapter = HTTPAdapter(
                        pool_connections=1,
                        pool_maxsize=1,
                        max_retries=Retry(
                            total=3,
                            backoff_factor=1,
                            status_forcelist=[408, 429, 500, 502, 503, 504],
                            allowed_methods=["GET", "POST"],
                        ),
                    )
                    session.mount("https://", adapter)

                    # Build the complete headers with security information
                    base_headers = self._build_header()
                    if headers:
                        # Merge user-provided headers with base headers
                        base_headers.update(headers)
                    headers = base_headers
                    headers["Host"] = hostname

                    # Add certificate info to request kwargs
                    request_kwargs = {
                        "method": method,
                        "url": url,
                        "headers": headers,
                        "data": data,
                        "params": params,
                        "stream": stream,
                        "verify": self.cert_verification,
                        "timeout": timeout or self.timeout,
                    }

                    # Add client certificate if mTLS is enabled
                    if self.require_client_auth:
                        if not self.certificate or not self.private_key:
                            raise ValueError(
                                "Both certificate and private key are required for mTLS "
                                "(client authentication). "
                                "Please provide both certificate and private key paths."
                            )
                        # Use proper cert format
                        request_kwargs["cert"] = (self.certificate, self.private_key)

                    response = session.request(**request_kwargs)
                else:
                    # For non-TLS requests, still use the security headers
                    base_headers = self._build_header()
                    if headers:
                        base_headers.update(headers)

                    response = session.request(
                        method=method,
                        url=url,
                        headers=base_headers,
                        data=data,
                        params=params,
                        stream=stream,
                        timeout=timeout or self.timeout,
                    )
                return response
            except requests.exceptions.SSLError as e:
                self._handle_ssl_error(e, attempt, max_retries)
                if attempt == max_retries - 1:
                    raise

    def _handle_ssl_error(self, e, attempt, max_retries):
        """Handle SSL errors with retry logic."""
        if "CERTIFICATE_UNKNOWN" in str(e):
            logger.error(
                "Certificate unknown error - this usually means the "
                "server's certificate is not trusted"
            )
            logger.error("Please verify that:")
            logger.error("1. The root certificate contains all necessary intermediate certificates")
            logger.error("2. The server's certificate is properly signed by a trusted CA")
            logger.error("3. The hostname matches the certificate's subject")
            if attempt < max_retries - 1 and self.refetch_server_cert_callback:
                logger.debug("Attempting to refetch server certificate")
                self.root_certificate = self.refetch_server_cert_callback()
                # Update the cert_verification with the new root certificate
                self.cert_verification = self._configure_cert_verification(
                    self.use_tls, self.root_certificate
                )
                # Re-verify certificates
                try:
                    self._verify_certificates()
                except Exception as verify_error:
                    logger.error(f"Certificate re-verification failed: {verify_error}")
                    raise
            else:
                raise
        else:
            if attempt < max_retries - 1:
                logger.warning(f"SSL error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(1)
            else:
                raise

    def _validate_response(self, response):
        """Validate response headers and security settings."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
        }
        for header, expected_value in security_headers.items():
            if header in response.headers and response.headers[header] != expected_value:
                logger.warning(f"Missing or incorrect security header: {header}")

        response.raise_for_status()

    def _handle_connection_error(self, e):
        """Handle connection errors."""
        logger.error(f"Connection error: {e}")
        if hasattr(e, "args") and len(e.args) > 0:
            logger.error(f"Connection error details: {e.args[0]}")

    def _handle_request_error(self, e):
        """Handle request errors."""
        logger.error(f"Request failed: {e}")
        if hasattr(e, "args") and len(e.args) > 0:
            logger.error(f"Request error details: {e.args[0]}")

    def get_tasks(self) -> Tuple[List[Any], int, int, bool]:
        """Get tasks from the aggregator with proper security settings."""
        headers = {"Accept": "application/json", "Sender": self.collaborator_name}
        params = {
            "collaborator_id": self.collaborator_name,
            "federation_uuid": self.federation_uuid,
        }
        url = f"{self.base_url}/tasks"
        response = self._make_request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        tasks_resp = aggregator_pb2.GetTasksResponse()
        json_format.ParseDict(data, tasks_resp)

        logger.debug(
            f"Received tasks response - Round: {tasks_resp.round_number}, "
            f"Tasks: {[t.name for t in tasks_resp.tasks]}, "
            f"Sleep: {tasks_resp.sleep_time}, Quit: {tasks_resp.quit}"
        )
        return tasks_resp.tasks, tasks_resp.round_number, tasks_resp.sleep_time, tasks_resp.quit

    def get_aggregated_tensor(
        self,
        tensor_name: str,
        round_number: int,
        report: bool,
        tags: List[str],
        require_lossless: bool,
    ) -> Any:
        """Get aggregated tensor with proper security settings."""
        params = {
            "sender": self.collaborator_name,
            "receiver": self.aggregator_uuid,
            "federation_uuid": self.federation_uuid,
            "tensor_name": tensor_name,
            "round_number": round_number,
            "report": report,
            "tags": tags,
            "require_lossless": require_lossless,
            "collaborator_id": self.collaborator_name,
        }
        headers = {"Accept": "application/json", "Sender": self.collaborator_name}
        url = f"{self.base_url}/tensors/aggregated"
        extended_timeout = (30, 600)  # 30 seconds connect, 10 minutes read timeout
        try:
            logger.debug(f"Requesting aggregated tensor {tensor_name} for round {round_number}")
            response = self._make_request(
                "GET", url, params=params, headers=headers, timeout=extended_timeout
            )
            data = response.json()
            resp = aggregator_pb2.GetAggregatedTensorResponse()
            json_format.ParseDict(data, resp, ignore_unknown_fields=True)
            logger.debug(f"Successfully retrieved tensor {tensor_name} for round {round_number}")
            return resp.tensor
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # This is expected during round 0 or when tensor hasn't been aggregated yet
                logger.debug(
                    f"No aggregated tensor found for {tensor_name} at round {round_number}"
                )
                return None
            raise

    def send_local_task_results(
        self,
        round_number: int,
        task_name: str,
        data_size: int,
        named_tensors: List[Any],
    ) -> bool:
        """Send local task results with proper security settings."""
        logger.debug(f"Sending task results for round {round_number}, task {task_name}")

        # Create the TaskResults message
        task_results = aggregator_pb2.TaskResults(
            header=aggregator_pb2.MessageHeader(
                sender=self.collaborator_name,
                receiver=self.aggregator_uuid,
                federation_uuid=self.federation_uuid,
                single_col_cert_common_name=self.single_col_cert_common_name or "",
            ),
            round_number=round_number,
            task_name=task_name,
            data_size=data_size,
            tensors=named_tensors,
        )

        # Serialize the TaskResults first
        task_results_bytes = task_results.SerializeToString()
        logger.debug(f"TaskResults serialized size: {len(task_results_bytes)} bytes")

        # Create a DataStream message containing the TaskResults bytes
        data_stream = base_pb2.DataStream(size=len(task_results_bytes), npbytes=task_results_bytes)

        # Create an empty DataStream to signal end of stream
        end_stream = base_pb2.DataStream(size=0, npbytes=b"")

        # Serialize both messages
        data_bytes = data_stream.SerializeToString()
        end_bytes = end_stream.SerializeToString()

        # Create length-prefixed stream format
        stream_data = (
            struct.pack(">I", len(data_bytes))  # Length prefix for first message
            + data_bytes  # First message
            + struct.pack(">I", len(end_bytes))  # Length prefix for second message
            + end_bytes  # Second message (empty message signals end)
        )

        url = f"{self.base_url}/tasks/results"
        request_headers = self._build_header()
        request_headers["Sender"] = self.collaborator_name
        request_headers["Content-Type"] = "application/x-protobuf-stream"
        request_headers["Content-Length"] = str(len(stream_data))

        try:
            response = self._make_request(
                "POST",
                url,
                data=stream_data,
                headers=request_headers,
                timeout=(30, 60),  # Keep shorter timeout since we're sending all data at once
            )
            response.raise_for_status()
            logger.debug(f"Successfully sent task results for round {round_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to send task results for round {round_number}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Request headers were: {request_headers}")
            raise

    def ping(self):
        """Ping the aggregator to check connectivity."""
        logger.info("Aggregator ping...")
        headers = {"Accept": "application/json", "Sender": self.collaborator_name}
        params = {
            "collaborator_id": self.collaborator_name,
            "federation_uuid": self.federation_uuid,
        }
        url = f"{self.base_url}/ping"
        response = self._make_request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Validate response header like GRPC client
        header = data.get("header", {})
        assert header.get("receiver") == self.collaborator_name, (
            f"Receiver in response header does not match collaborator name. "
            f"Expected: {self.collaborator_name}, Actual: {header.get('receiver')}"
        )
        assert header.get("sender") == self.aggregator_uuid, (
            f"Sender in response header does not match aggregator UUID. "
            f"Expected: {self.aggregator_uuid}, Actual: {header.get('sender')}"
        )
        assert header.get("federationUuid") == self.federation_uuid, (
            f"Federation UUID in response header does not match. "
            f"Expected: {self.federation_uuid}, Actual: {header.get('federationUuid')}"
        )
        assert header.get("singleColCertCommonName", "") == (
            self.single_col_cert_common_name or ""
        ), (
            f"Single collaborator certificate common name in response header does not match. "
            f"Expected: {self.single_col_cert_common_name}, "
            f"Actual: {header.get('singleColCertCommonName')}"
        )

        logger.info("Aggregator pong!")

    def send_message_to_server(self, openfl_message: Any, collaborator_name: str) -> Any:
        """
        Forwards a converted message from the local REST client to the OpenFL server and returns
        the response.

        Args:
            openfl_message: The InteropMessage proto to be sent to the OpenFL server.
            collaborator_name: The name of the collaborator.

        Returns:
            The response from the OpenFL server (InteropMessage proto).
        """
        # Set the header fields
        header = aggregator_pb2.MessageHeader(
            sender=collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name or "",
        )
        openfl_message.header.CopyFrom(header)

        # Serialize to JSON
        json_payload = json_format.MessageToJson(openfl_message)
        url = f"{self.base_url}/interop/relay"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Sender": collaborator_name,
        }
        response = self._make_request(
            "POST",
            url,
            data=json_payload,
            headers=headers,
            timeout=(30, 300),
        )
        response.raise_for_status()
        response_json = response.json()
        openfl_response = aggregator_pb2.InteropMessage()
        json_format.ParseDict(response_json, openfl_response, ignore_unknown_fields=True)
        return openfl_response

    def __del__(self):
        """Cleanup when the client is destroyed."""
        self.session.close()
