# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AggregatorRESTServer module."""

import logging
import ssl
import threading
import time
from functools import wraps
from random import random
from time import sleep

from flask import Flask, abort, jsonify, request
from google.protobuf import json_format
from werkzeug.serving import make_server

from openfl.protocols import aggregator_pb2, base_pb2

logger = logging.getLogger(__name__)


def synchronized(func):
    """Synchronization decorator."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def create_header(sender, receiver, federation_uuid, single_col_cert_common_name=""):
    """Create a standard message header with consistent fields."""
    return aggregator_pb2.MessageHeader(
        sender=str(sender),
        receiver=str(receiver),
        federation_uuid=str(federation_uuid),
        single_col_cert_common_name=single_col_cert_common_name or "",
    )


class AggregatorRESTServer:
    """REST server for the aggregator."""

    def __init__(
        self,
        aggregator,
        agg_addr,
        agg_port,
        use_tls=True,
        require_client_auth=True,
        certificate=None,
        private_key=None,
        root_certificate=None,
        **kwargs,
    ):
        """Initialize REST server with security defaults."""
        # Initialize lock for synchronized methods
        self._lock = threading.Lock()

        # Set up base configuration
        self.aggregator = aggregator
        self.host = agg_addr
        self.port = agg_port

        # Set API prefix
        self.api_prefix = "experimental/v1"

        # Set security defaults
        self.use_tls = use_tls
        self.require_client_auth = require_client_auth
        self.ssl_context = None

        # Set up server components with security focus
        self._setup_server_components(certificate, private_key, root_certificate)

        # Set up routes with synchronized access
        self._setup_routes()

        # Build the base URL
        scheme = "https" if use_tls else "http"
        self.base_url = f"{scheme}://{agg_addr}:{agg_port}/{self.api_prefix}"

    def _setup_ssl_context(self, certificate, private_key, root_certificate):
        """Set up SSL context for TLS/mTLS."""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Set secure cipher suites
        ssl_context.set_ciphers("ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384")

        # Disable older TLS versions and set security options
        ssl_context.options |= (
            ssl.OP_NO_TLSv1
            | ssl.OP_NO_TLSv1_1
            | ssl.OP_NO_TLSv1_2
            | ssl.OP_NO_COMPRESSION
            | ssl.OP_NO_TICKET  # Disable session tickets
            | ssl.OP_CIPHER_SERVER_PREFERENCE  # Server chooses cipher
            | ssl.OP_SINGLE_DH_USE  # Ensure perfect forward secrecy with DHE
            | ssl.OP_SINGLE_ECDH_USE  # Ensure perfect forward secrecy with ECDHE
        )

        # Set verification flags for strict certificate checking
        ssl_context.verify_flags = (
            ssl.VERIFY_X509_STRICT | ssl.VERIFY_CRL_CHECK_CHAIN  # Check certificate revocation
        )

        # Configure client certificate verification
        if self.require_client_auth:
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            # Load root CA for client cert verification
            if root_certificate:
                try:
                    ssl_context.load_verify_locations(cafile=root_certificate)
                except Exception as e:
                    logger.error(f"Failed to load root CA certificate: {str(e)}")
                    raise
            else:
                logger.error("Root certificate is required when client authentication is enabled")
                raise ValueError("Root certificate is required for mTLS")
        else:
            ssl_context.verify_mode = ssl.CERT_NONE

        # Load server certificate and key
        try:
            ssl_context.load_cert_chain(certfile=certificate, keyfile=private_key)
        except Exception as e:
            logger.error(f"Failed to load server certificate and key: {str(e)}")
            raise

        # Load and trust the root CA certificate
        if root_certificate:
            try:
                ssl_context.load_verify_locations(cafile=root_certificate)
            except Exception as e:
                logger.error(f"Failed to load root CA certificate: {str(e)}")
                raise
        # Enable post-handshake authentication for better security
        if hasattr(ssl_context, "post_handshake_auth"):
            ssl_context.post_handshake_auth = True

        # Set verification purpose
        ssl_context.purpose = ssl.Purpose.CLIENT_AUTH

        return ssl_context

    def _setup_flask_app(self):
        """Configure Flask application with proper settings for both TLS and non-TLS modes."""
        app = Flask(__name__)

        # Set session and file age defaults
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1800  # 30 minutes
        app.config["PERMANENT_SESSION_LIFETIME"] = 1800  # 30 minutes

        # Configure logging to be minimal
        import logging

        # Disable Flask's default logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        # Add security headers
        @app.after_request
        def add_security_headers(response):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            if self.use_tls:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )
            return response

        return app

    def _validate_client_certificate(self, request_environ, collaborator_name):
        """
        Validate client certificate when mTLS is enabled.
        Args:
            request_environ: The request environment containing SSL information
            collaborator_name: The collaborator name from the request (from header.sender)
        Returns:
            bool: True if validation passes
        Raises:
            abort: HTTP error if validation fails
        """
        if not self.use_tls:
            return True

        try:
            # Default to collaborator name (like gRPC)
            common_name = collaborator_name

            # Get certificate information if client auth is required
            if self.require_client_auth:
                cert_cn = self._get_certificate_cn(request_environ, collaborator_name)
                if not cert_cn:
                    abort(401, "Client certificate validation failed - certificate not found")
                common_name = cert_cn

            # Validate collaborator identity
            return self._validate_collaborator(common_name, collaborator_name)

        except Exception as e:
            logger.error(f"Certificate validation failed: {str(e)}")
            abort(401, str(e))

    def _get_certificate_cn(self, request_environ, collaborator_name):
        """Get certificate CN from environment or headers."""
        # Try to get certificate info from environment
        peercert = request_environ.get("SSL_CLIENT_CERT")
        cert_cn = request_environ.get("SSL_CLIENT_S_DN_CN")

        # Try to extract CN if we have certificate but no CN
        if peercert and not cert_cn:
            try:
                cert_cn = self._extract_cn_from_cert(peercert)
            except Exception as e:
                logger.error(f"Failed to extract CN from certificate: {e}")

        # If no certificate found, try fallback methods
        if not peercert:
            # Try header-based fallback for experimental mode
            cert_cn = self._try_header_fallback(collaborator_name)

        return cert_cn

    def _try_header_fallback(self, collaborator_name):
        """Try to get CN from headers as fallback in experimental mode."""
        # FALLBACK: In experimental mode, allow using header-based auth
        # This should NOT be used in production
        try:
            from flask import request

            # Use Sender header as fallback
            if hasattr(request, "headers") and "Sender" in request.headers:
                cert_cn = request.headers.get("Sender")
                return cert_cn
        except Exception as e:
            logger.error(f"Error in header fallback: {e}")

        # THIS SHOULD BE REMOVED POST EXPERIMENTAL MODE
        return collaborator_name

    def _validate_collaborator(self, common_name, collaborator_name):
        """Validate collaborator identity."""
        if not self.aggregator.valid_collaborator_cn_and_id(common_name, collaborator_name):
            # Add timing attack protection
            sleep(5 * random())
            logger.error(
                f"Invalid collaborator. CN: |{common_name}| "
                f"collaborator_name: |{collaborator_name}|"
            )
            abort(401, "Collaborator validation failed")

        return True

    def _extract_cn_from_cert(self, cert_pem):
        """Extract CN from a PEM certificate using standard libraries."""
        import re

        pass
        pass

        # Try regex approach first (most reliable with PEM format)
        cn_match = re.search(
            r"CN\s*=\s*([^,/\n]+)",
            cert_pem.decode("utf-8") if isinstance(cert_pem, bytes) else cert_pem,
        )
        if cn_match:
            return cn_match.group(1).strip()

        # Try using cryptography if available
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend

            # Convert PEM to certificate object
            cert_data = cert_pem.encode("utf-8") if isinstance(cert_pem, str) else cert_pem
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())

            # Extract CN from subject
            for attribute in cert.subject:
                if attribute.oid._name == "commonName":
                    return attribute.value
        except ImportError:
            pass

        # Fallback: use the collaborator name from the environment
        return None

    def _setup_interop_client(self):
        """Set up inter-federation connector client."""
        try:
            return self.aggregator.get_interop_client()
        except AttributeError:
            return None

    def _is_authorized(self, collaborator_id, federation_id, cert_common_name=None):
        """
        Validate collaborator identity with strict checks.

        Args:
            collaborator_id (str): The collaborator's ID
            federation_id (str): The federation UUID
            cert_common_name (str, optional): Certificate CN if using mTLS

        Returns:
            bool: True if validation passes

        Raises:
            abort: HTTP error if validation fails
        """
        is_valid = False
        try:
            # Validate collaborator identity
            if not collaborator_id:
                logger.error("Collaborator identity not provided")
                abort(400, "Collaborator identity not provided")

            # First check if collaborator is authorized
            if collaborator_id not in self.aggregator.authorized_cols:
                logger.error(f"Collaborator not in authorized list. Got: {collaborator_id}")
                abort(401, "Unauthorized collaborator")

            # Validate collaborator identity
            common_name = cert_common_name if cert_common_name is not None else collaborator_id
            if not self.aggregator.valid_collaborator_cn_and_id(common_name, collaborator_id):
                logger.error(
                    f"Collaborator validation failed. CN: {common_name}, ID: {collaborator_id}"
                )
                abort(401, "Collaborator validation failed")

            # Validate client certificate if mTLS is enabled
            if self.use_tls and self.require_client_auth:
                self._validate_client_certificate(request.environ, collaborator_id)

            # Verify federation UUID
            if federation_id != str(self.aggregator.federation_uuid):
                logger.error(
                    f"Federation UUID mismatch. Expected: {self.aggregator.federation_uuid}, "
                    f"Got: {federation_id}"
                )
                abort(401, "Federation UUID mismatch")

            is_valid = True
            return True

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            abort(401, str(e))
        finally:
            # Add timing attack protection for all error cases
            if not is_valid:
                sleep(5 * random())

    def _validate_task_headers(self, headers):
        """
        Validate task submission headers with timing attack protection.

        Args:
            headers (dict): Request headers

        Returns:
            str: Validated collaborator name

        Raises:
            abort: HTTP error if validation fails
        """
        try:
            # Get collaborator identity from certificate or headers
            collab_name = None
            if self.use_tls and self.require_client_auth:
                # Try to get from certificate first
                collab_name = request.environ.get("SSL_CLIENT_S_DN_CN")
                logger.debug(f"Using certificate CN: {collab_name}")

            # If not from certificate, try headers
            if not collab_name:
                collab_name = headers.get("Sender")
                if not collab_name:
                    sleep(5 * random())  # Add timing attack protection
                    logger.error("No Sender header provided")
                    abort(401, "No Sender header provided")

            # Get other required headers
            receiver = headers.get("Receiver")
            federation_id = headers.get("Federation-UUID")
            cert_common_name = headers.get("Single-Col-Cert-CN", "")

            # Validate collaborator identity
            if not self.aggregator.valid_collaborator_cn_and_id(collab_name, collab_name):
                sleep(5 * random())  # Add timing attack protection
                msg = f"CN: {collab_name}, ID: {collab_name}"
                logger.error(f"Collaborator validation failed. {msg}")
                abort(401, "Collaborator validation failed")

            # Verify all headers with strict validation
            assert receiver == str(self.aggregator.uuid), (
                f"Header receiver mismatch. Expected: {self.aggregator.uuid}, Got: {receiver}"
            )

            assert federation_id == str(self.aggregator.federation_uuid), (
                f"Federation UUID mismatch. Expected: {self.aggregator.federation_uuid}, "
                f"Got: {federation_id}"
            )

            expected_cn = self.aggregator.single_col_cert_common_name or ""
            assert cert_common_name == expected_cn, (
                f"Single col cert CN mismatch. Expected: {expected_cn}, Got: {cert_common_name}"
            )

            return collab_name
        except AssertionError as e:
            sleep(5 * random())  # Add timing attack protection
            logger.error(f"Header validation failed: {str(e)}")
            abort(401, str(e))

    def _parse_protobuf_stream(self, data):
        """Parse protobuf stream data."""
        logger.debug(f"Received {len(data)} bytes of protobuf stream data")

        # First message is DataStream containing TaskResults
        msg_len = int.from_bytes(data[:4], byteorder="big")
        logger.debug(f"First message length: {msg_len}")
        data_stream_bytes = data[4 : 4 + msg_len]
        data_stream = base_pb2.DataStream()
        data_stream.ParseFromString(data_stream_bytes)
        logger.debug(f"Parsed DataStream with size: {data_stream.size}")

        # Extract TaskResults from DataStream
        task_results = aggregator_pb2.TaskResults()
        task_results.ParseFromString(data_stream.npbytes)

        # Log task details
        task_info = (
            f"Task: {task_results.task_name}, "
            f"Round: {task_results.round_number}, "
            f"Size: {task_results.data_size}, "
            f"Tensors: {len(task_results.tensors)}"
        )
        logger.debug(f"Extracted TaskResults from DataStream - {task_info}")

        # Verify end message
        end_msg_offset = 4 + msg_len
        end_msg_len = int.from_bytes(data[end_msg_offset : end_msg_offset + 4], byteorder="big")
        logger.debug(f"End message length: {end_msg_len}")

        if end_msg_len != 0:
            logger.error(f"Invalid end message length: {end_msg_len}")
            abort(400, "Invalid stream format - expected empty end message")

        # Verify total length
        expected_total_len = 4 + msg_len + 4 + end_msg_len
        if len(data) != expected_total_len:
            msg = f"Got {len(data)}, expected {expected_total_len}"
            logger.error(f"Data length mismatch. {msg}")
            abort(400, "Invalid stream data length")

        return task_results

    def _build_tasks_response(
        self,
        tasks_list,
        round_number,
        sleep_time,
        time_to_quit,
        collab_id,
    ):
        """Build GetTasksResponse protobuf."""
        tasks_proto = []
        if tasks_list:
            if isinstance(tasks_list[0], str):
                # Backward compatibility: list of task names
                tasks_proto = [aggregator_pb2.Task(name=t) for t in tasks_list]
            else:
                tasks_proto = [
                    aggregator_pb2.Task(
                        name=getattr(t, "name", ""),
                        function_name=getattr(t, "function_name", ""),
                        task_type=getattr(t, "task_type", ""),
                        apply_local=getattr(t, "apply_local", False),
                    )
                    for t in tasks_list
                ]

        # Create response header
        header = create_header(
            sender=str(self.aggregator.uuid),
            receiver=collab_id,
            federation_uuid=str(self.aggregator.federation_uuid),
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name or "",
        )

        return aggregator_pb2.GetTasksResponse(
            header=header,
            round_number=round_number,
            tasks=tasks_proto,
            sleep_time=sleep_time,
            quit=time_to_quit,
        )

    def _setup_server_components(self, certificate=None, private_key=None, root_certificate=None):
        """Set up server components including SSL, Flask app, and interop client."""
        # Set up SSL if enabled
        if self.use_tls:
            self.ssl_context = self._setup_ssl_context(certificate, private_key, root_certificate)
        else:
            self.ssl_context = None  # Explicitly set to None when TLS is disabled

        # Set up Flask app
        self.app = self._setup_flask_app()

        # Set up interop client
        self.interop_client = self._setup_interop_client()
        self.use_connector = self.interop_client is not None

    def _setup_routes(self):
        """Set up Flask routes."""
        # Register the route handlers
        self._setup_ping_route()
        self._setup_tasks_route()
        self._setup_task_results_route()
        self._setup_tensor_route()
        self._setup_relay_route()
        # Add middleware for client certificate extraction
        self._setup_certificate_middleware()

    def _setup_certificate_middleware(self):
        """Set up middleware to capture SSL certificates from client connections."""

        @self.app.before_request
        def extract_client_cert():
            """Extract client certificate and add it to request environment."""
            if not (self.use_tls and self.require_client_auth):
                return None

            # Get SSL connection information
            try:
                from flask import request

                # Try to extract certificate from the socket
                cert_data = self._extract_certificate_from_socket(request.environ)
                if cert_data:
                    # Process the certificate data
                    self._process_certificate_data(request.environ, cert_data)

            except Exception as e:
                logger.warning(f"Failed to extract client certificate: {e}")
                # Continue processing the request even if cert extraction fails

            return None

    def _extract_certificate_from_socket(self, environ):
        """Extract the certificate from the socket if available."""
        # Access underlying SSL socket if possible
        transport = environ.get("werkzeug.socket")
        if not (transport and hasattr(transport, "getpeercert")):
            return None

        # Extract certificate from socket
        return transport.getpeercert(binary_form=True)

    def _process_certificate_data(self, environ, der_cert):
        """Process the DER certificate data and store in environment."""
        if not der_cert:
            return False

        # Convert DER to PEM format using built-in libraries
        try:
            # Try using cryptography if available
            cn, pem_cert = self._convert_der_using_cryptography(der_cert)
            if pem_cert:
                environ["SSL_CLIENT_CERT"] = pem_cert
            if cn:
                environ["SSL_CLIENT_S_DN_CN"] = cn
                logger.info(f"Extracted client certificate CN: {cn}")
                return True
        except ImportError:
            # Fall back to regex method
            return self._try_regex_cn_extraction(environ, der_cert)
        except Exception as e:
            logger.warning(f"Error converting certificate format: {e}")

        return False

    def _convert_der_using_cryptography(self, der_cert):
        """Convert DER certificate using cryptography library."""
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        cert = x509.load_der_x509_certificate(der_cert, default_backend())
        pem_cert = cert.public_bytes(encoding=serialization.Encoding.PEM)

        # Parse the subject to get CN
        cn = None
        for attribute in cert.subject:
            if attribute.oid._name == "commonName":
                cn = attribute.value
                break

        return cn, pem_cert

    def _try_regex_cn_extraction(self, environ, der_cert):
        """Try to extract CN using regex from binary certificate."""
        try:
            import binascii
            import re

            # Convert to hex and then look for CN
            hex_data = binascii.hexlify(der_cert).decode("ascii")
            # Look for common name pattern in hex
            # This is a simplified approach and may not work for all certs
            cn_pattern = (
                r"(?:3[0-9]|4[0-9]|5[0-9])(?:06|07|08|09|0a|0b|0c|0d|0e|0f)"
                r"(?:03|04|05|06)(?:13|14|15|16)(.{2,60})(?:30|31)"
            )
            cn_match = re.search(cn_pattern, hex_data)
            if cn_match:
                # Convert hex to ASCII
                cn_hex = cn_match.group(1)
                try:
                    cn = binascii.unhexlify(cn_hex).decode("utf-8")
                    environ["SSL_CLIENT_S_DN_CN"] = cn
                    logger.info(f"Extracted client certificate CN using regex: {cn}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to decode CN: {e}")

            return False
        except Exception as e:
            logger.warning(f"Error in regex CN extraction: {e}")
            return False

    def _setup_ping_route(self):
        """Set up the /ping endpoint."""

        @self.app.route(f"/{self.api_prefix}/ping", methods=["GET"])
        def ping():
            """Simple ping endpoint to check server connectivity."""
            try:
                # Get collaborator identity from certificate or query param
                collaborator_id = None
                if self.require_client_auth:
                    collaborator_id = request.environ.get("SSL_CLIENT_S_DN_CN")
                if collaborator_id is None:
                    collaborator_id = request.args.get("collaborator_id")

                federation_id = request.args.get("federation_uuid")

                # Use the consolidated validation method
                self._is_authorized(collaborator_id, federation_id)

                # Create response header
                header = create_header(
                    sender=str(self.aggregator.uuid),
                    receiver=collaborator_id,
                    federation_uuid=str(self.aggregator.federation_uuid),
                    single_col_cert_common_name=self.aggregator.single_col_cert_common_name or "",
                )

                # Return response in same format as GRPC
                return jsonify({"header": json_format.MessageToDict(header)})
            except Exception as e:
                logger.error(f"Ping request failed: {str(e)}")
                abort(401, str(e))

    def _setup_tasks_route(self):
        """Set up the /tasks endpoint."""

        @self.app.route(f"/{self.api_prefix}/tasks", methods=["GET"])
        def get_tasks():
            """Endpoint for collaborators to fetch pending tasks."""
            # Get collaborator identity from certificate or query param
            collaborator_id = None
            if self.require_client_auth:
                collaborator_id = request.environ.get("SSL_CLIENT_S_DN_CN")
            if collaborator_id is None:
                collaborator_id = request.args.get("collaborator_id")

            federation_id = request.args.get("federation_uuid")

            # Use the consolidated validation method
            self._is_authorized(collaborator_id, federation_id)

            # Check if connector mode is enabled
            if self.use_connector:
                abort(501, "GetTasks not supported in connector mode")

            # Fetch tasks from Aggregator core - directly delegate to the aggregator
            tasks_list, round_number, sleep_time, time_to_quit = self.aggregator.get_tasks(
                collaborator_id
            )

            # Log task assignment
            task_names = [getattr(t, "name", t) for t in (tasks_list or [])]
            logger.debug(
                f"Collaborator {collaborator_id} requested tasks. "
                f"Round: {round_number}, Tasks: {task_names}, "
                f"Sleep: {sleep_time}, Quit: {time_to_quit}"
            )

            # Build and return response
            response_proto = self._build_tasks_response(
                tasks_list, round_number, sleep_time, time_to_quit, collaborator_id
            )
            return jsonify(json_format.MessageToDict(response_proto))

    def _setup_task_results_route(self):
        """Set up the /tasks/results endpoint."""

        @self.app.route(f"/{self.api_prefix}/tasks/results", methods=["POST"])
        def post_task_results():
            """Handle task results submission."""
            try:
                # Validate headers and get collaborator name
                collab_name = self._validate_task_headers(request.headers)

                # Parse protobuf stream data
                task_results = self._parse_protobuf_stream(request.data)

                # Direct delegation to the aggregator for task results processing
                # This matches the gRPC approach of calling send_local_task_results directly
                self.aggregator.send_local_task_results(
                    collab_name,
                    task_results.round_number,
                    task_results.task_name,
                    task_results.data_size,
                    task_results.tensors,
                )

                return jsonify({"status": "success"})

            except Exception as e:
                logger.error(f"Error processing task results: {str(e)}")
                abort(400, f"Error processing task results: {str(e)}")

    def _setup_tensor_route(self):
        """Set up the /tensors/aggregated endpoint."""

        @self.app.route(f"/{self.api_prefix}/tensors/aggregated", methods=["GET"])
        def get_aggregated_tensor():
            """Endpoint for collaborators to retrieve an aggregated tensor."""
            start_time = time.time()

            # Validate that this endpoint is not used in connector mode
            if self.use_connector:
                abort(501, "GetAggregatedTensor not supported in connector mode")

            # Get and validate collaborator identity
            collaborator_id = request.args.get("collaborator_id")
            federation_id = request.args.get("federation_uuid")

            # Use the consolidated validation method
            self._is_authorized(collaborator_id, federation_id)

            # Extract tensor request parameters
            tensor_name = request.args.get("tensor_name")
            try:
                round_number = int(request.args.get("round_number", 0))
            except (TypeError, ValueError):
                abort(400, "Invalid round number")
            report = request.args.get("report", "").lower() == "true"
            tags = request.args.getlist("tags")
            require_lossless = request.args.get("require_lossless", "").lower() == "true"

            # Get the tensor from aggregator - direct delegation to the aggregator
            named_tensor = self.aggregator.get_aggregated_tensor(
                tensor_name,
                round_number,
                report=report,
                tags=tuple(tags),
                require_lossless=require_lossless,
                requested_by=collaborator_id,
            )

            # Create response header using the standardized method
            header = create_header(
                sender=str(self.aggregator.uuid),
                receiver=collaborator_id,
                federation_uuid=str(self.aggregator.federation_uuid),
                single_col_cert_common_name=self.aggregator.single_col_cert_common_name or "",
            )

            # Create response with empty tensor if not found
            response_proto = aggregator_pb2.GetAggregatedTensorResponse(
                header=header,
                round_number=round_number,
                tensor=named_tensor
                if named_tensor is not None
                else aggregator_pb2.NamedTensorProto(),
            )

            logger.debug(f"Tensor retrieval completed in {time.time() - start_time:.2f} seconds")
            return jsonify(json_format.MessageToDict(response_proto))

    def _setup_relay_route(self):
        """Set up the /interop/relay endpoint."""

        @self.app.route(f"/{self.api_prefix}/interop/relay", methods=["POST"])
        def relay_message():
            """Endpoint for collaborator-to-aggregator message relay."""
            # This endpoint is optional; only enable if connector mode is configured
            if not self.use_connector or self.interop_client is None:
                abort(501, "Interop relay is not enabled on this aggregator")

            # Parse the incoming JSON to an InteropRelay protobuf message
            try:
                relay_req = json_format.Parse(
                    request.data.decode("utf-8"), aggregator_pb2.InteropRelay()
                )
            except Exception as e:
                abort(400, f"Invalid InteropRelay payload: {e}")

            # Validate the collaborator via header
            collab_name = relay_req.header.sender
            self._is_authorized(collab_name, relay_req.header.federation_uuid)

            if relay_req.header.receiver != str(self.aggregator.uuid):
                abort(400, "Header receiver mismatch")

            # Forward the request to the configured interop connector and get response
            logger.debug(
                f"Relaying message from {collab_name} to external federation via connector"
            )
            # Create a header for forwarding using the standardized method
            forward_header = create_header(
                sender=str(self.aggregator.uuid),
                receiver=relay_req.header.receiver,
                federation_uuid=str(self.aggregator.federation_uuid),
                single_col_cert_common_name=self.aggregator.single_col_cert_common_name or "",
            )
            # Use the aggregator's interop client to send and receive
            response_proto = self.interop_client.send_receive(relay_req, header=forward_header)
            # Return the response from the remote as JSON
            return jsonify(json_format.MessageToDict(response_proto))

    def serve(self):
        """Start the REST server with proper configuration for both TLS and non-TLS modes."""
        # If connector mode is enabled, start the connector service
        if self.use_connector:
            try:
                self.aggregator.start_connector()
            except AttributeError:
                pass

        # Configure server based on TLS mode
        if self.use_tls and self.ssl_context:
            server = make_server(
                self.host,
                self.port,
                self.app,
                ssl_context=self.ssl_context,
                threaded=True,  # Enable threading for better performance
            )
        else:
            server = make_server(
                self.host,
                self.port,
                self.app,
                threaded=True,  # Enable threading for better performance
            )

        # Configure server thread
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        logger.warning(
            "Starting Aggregator REST Server (EXPERIMENTAL API - Not for production use)"
        )
        thread.start()

        try:
            while not self.aggregator.all_quit_jobs_sent():
                sleep(5)
        finally:
            # Synchronized shutdown
            if self.use_connector:
                try:
                    self.aggregator.stop_connector()
                except AttributeError:
                    pass
            server.shutdown()
            thread.join()
            logger.info("Aggregator REST Server stopped.")
