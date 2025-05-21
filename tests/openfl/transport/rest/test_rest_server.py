# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""REST server tests module."""

import pytest
import ssl
from unittest import mock
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta

from openfl.transport.rest.aggregator_server import AggregatorRESTServer
from openfl.protocols import aggregator_pb2, base_pb2


def generate_test_certificates(cert_path, key_path, root_cert_path):
    """Generate self-signed certificates for testing."""
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    # Generate self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"test.example.com"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Test Organization"),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=1)
    ).sign(private_key, hashes.SHA256())

    # Write private key
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # For testing, use the same cert as root CA
    with open(root_cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


@pytest.fixture
def mock_aggregator():
    """Create a mock aggregator for testing."""
    aggregator = mock.Mock()
    aggregator.uuid = "test-uuid"
    aggregator.federation_uuid = "fed-uuid"
    aggregator.authorized_cols = ["test-collaborator"]
    aggregator.single_col_cert_common_name = "test-cert-cn"
    aggregator.valid_collaborator_cn_and_id = mock.Mock(return_value=True)
    aggregator.get_tasks = mock.Mock(return_value=(["task1", "task2"], 1, 5, False))
    aggregator.get_aggregated_tensor = mock.Mock()
    aggregator.send_local_task_results = mock.Mock()
    # Disable connector mode by default
    aggregator.get_interop_client = mock.Mock(return_value=None)
    # Add mock for task completion tracking
    aggregator._collaborator_task_completed = mock.Mock(return_value=True)
    # Add mock assigner
    mock_assigner = mock.Mock()
    mock_assigner.get_tasks_for_collaborator = mock.Mock(return_value=[])
    aggregator.assigner = mock_assigner
    # Add collaborators_done list
    aggregator.collaborators_done = []
    return aggregator


@pytest.fixture
def ssl_certs(tmp_path):
    """Create temporary SSL certificate files for testing."""
    cert_path = tmp_path / "test_cert.pem"
    key_path = tmp_path / "test_key.pem"
    root_path = tmp_path / "test_root.pem"

    generate_test_certificates(cert_path, key_path, root_path)

    return {
        'cert': str(cert_path),
        'key': str(key_path),
        'root': str(root_path)
    }


@pytest.fixture
def rest_server(mock_aggregator, ssl_certs):
    """Create REST server instance for testing."""
    server = AggregatorRESTServer(
        aggregator=mock_aggregator,
        agg_addr="localhost",
        agg_port=8080,
        use_tls=True,
        require_client_auth=True,
        certificate=ssl_certs['cert'],
        private_key=ssl_certs['key'],
        root_certificate=ssl_certs['root']
    )
    return server


class TestAggregatorRESTServer:
    """Test cases for AggregatorRESTServer."""

    def test_ssl_context_setup(self, rest_server, ssl_certs):
        """Test SSL context configuration."""
        with mock.patch('ssl.SSLContext') as mock_ssl_context:
            mock_context = mock.Mock()
            mock_ssl_context.return_value = mock_context
            # Mock the options attribute to be an integer that can handle bitwise operations
            mock_context.options = 0

            rest_server._setup_ssl_context(
                certificate=ssl_certs['cert'],
                private_key=ssl_certs['key'],
                root_certificate=ssl_certs['root']
            )

            mock_ssl_context.assert_called_once_with(ssl.PROTOCOL_TLS_SERVER)

            # Check that load_cert_chain was called exactly once with the expected parameters
            mock_context.load_cert_chain.assert_called_once_with(
                certfile=ssl_certs['cert'],
                keyfile=ssl_certs['key']
            )

            # Check that load_verify_locations was called exactly twice with the same parameters
            assert mock_context.load_verify_locations.call_count == 2
            assert all(
                call == mock.call(cafile=ssl_certs['root'])
                for call in mock_context.load_verify_locations.call_args_list
            )

            assert mock_context.verify_mode == ssl.CERT_REQUIRED

    def test_get_tasks_valid_request(self, rest_server, mock_aggregator):
        """Test successful task retrieval."""
        # Mock the get_tasks method to return proper Task objects
        mock_tasks = [
            aggregator_pb2.Task(name="task1", function_name="func1", task_type="train"),
            aggregator_pb2.Task(name="task2", function_name="func2", task_type="validate")
        ]
        mock_aggregator.get_tasks.return_value = (mock_tasks, 1, 5, True)  # Set quit to True to ensure it appears in JSON

        with rest_server.app.test_client() as client:
            response = client.get('experimental/v1/tasks', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "fed-uuid"
            })

            assert response.status_code == 200
            data = response.get_json()
            assert data["roundNumber"] == 1
            assert len(data["tasks"]) == 2
            assert data["sleepTime"] == 5
            assert "quit" in data  # Verify quit field exists
            assert data["quit"]  # Should be True now

        # Test with quit=False
        mock_aggregator.get_tasks.return_value = (mock_tasks, 1, 5, False)

        with rest_server.app.test_client() as client:
            response = client.get('experimental/v1/tasks', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "fed-uuid"
            })

            assert response.status_code == 200
            data = response.get_json()
            # When quit is False (default value), it might be omitted in JSON
            # So we use get() with a default value
            assert not data.get("quit", False)

    def test_get_tasks_unauthorized(self, rest_server):
        """Test task retrieval with unauthorized collaborator."""
        with rest_server.app.test_client() as client:
            response = client.get('experimental/v1/tasks', query_string={
                "collaborator_id": "unauthorized-collaborator",
                "federation_uuid": "fed-uuid"
            })
            assert response.status_code == 401

    def test_post_task_results(self, rest_server, mock_aggregator):
        """Test task results submission."""
        # Create mock task results
        task_results = aggregator_pb2.TaskResults()
        task_results.task_name = "test_task"
        task_results.round_number = 1
        task_results.data_size = 100

        # Create mock header
        task_results.header.sender = "test-collaborator"
        task_results.header.receiver = str(mock_aggregator.uuid)
        task_results.header.federation_uuid = str(mock_aggregator.federation_uuid)
        task_results.header.single_col_cert_common_name = "test-cert-cn"

        # Add a named tensor
        tensor = base_pb2.NamedTensor()
        tensor.name = "test_tensor"
        task_results.tensors.append(tensor)

        # Create DataStream
        data_stream = base_pb2.DataStream()
        data_stream.npbytes = task_results.SerializeToString()
        data_stream.size = len(data_stream.npbytes)

        # Prepare request data
        request_data = (
            len(data_stream.SerializeToString()).to_bytes(4, byteorder='big') +
            data_stream.SerializeToString() +
            (0).to_bytes(4, byteorder='big')
        )

        # Configure mock assigner to return tasks
        mock_aggregator.assigner.get_tasks_for_collaborator.return_value = [
            aggregator_pb2.Task(name="test_task")
        ]

        with rest_server.app.test_client() as client:
            response = client.post(
                'experimental/v1/tasks/results',
                data=request_data,
                headers={
                    "Sender": "test-collaborator",
                    "Receiver": str(mock_aggregator.uuid),
                    "Federation-UUID": str(mock_aggregator.federation_uuid),
                    "Single-Col-Cert-CN": "test-cert-cn"
                }
            )

            assert response.status_code == 200
            mock_aggregator.send_local_task_results.assert_called_once()

    def test_get_aggregated_tensor(self, rest_server, mock_aggregator):
        """Test aggregated tensor retrieval."""
        # Create mock tensor response
        mock_tensor = base_pb2.NamedTensor()
        mock_tensor.name = "test_tensor"
        mock_aggregator.get_aggregated_tensor.return_value = mock_tensor

        with rest_server.app.test_client() as client:
            response = client.get('/experimental/v1/tensors/aggregated', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "fed-uuid",
                "tensor_name": "test_tensor",
                "round_number": "1"
            })

            assert response.status_code == 200
            data = response.get_json()
            assert data["roundNumber"] == 1
            assert "tensor" in data

    def test_relay_message_not_enabled(self, rest_server):
        """Test relay endpoint when not enabled."""
        # Create a valid relay message
        relay_msg = aggregator_pb2.InteropMessage()
        relay_msg.header.sender = "test-collaborator"
        relay_msg.header.receiver = str(rest_server.aggregator.uuid)
        relay_msg.header.federation_uuid = str(rest_server.aggregator.federation_uuid)

        with rest_server.app.test_client() as client:
            response = client.post(
                '/experimental/v1/interop/relay',
                json={"header": {"sender": "test-collaborator"}}
            )
            assert response.status_code == 501

    def test_invalid_federation_uuid(self, rest_server):
        """Test request with invalid federation UUID."""
        with rest_server.app.test_client() as client:
            response = client.get('/experimental/v1/tasks', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "invalid-uuid"
            })
            assert response.status_code == 401

    def test_malformed_task_results(self, rest_server):
        """Test submission of malformed task results."""
        with rest_server.app.test_client() as client:
            response = client.post(
                'experimental/v1/tasks/results',
                data=b"invalid data",
                headers={
                    "Sender": "test-collaborator",
                    "Receiver": str(rest_server.aggregator.uuid),
                    "Federation-UUID": str(rest_server.aggregator.federation_uuid)
                }
            )
            assert response.status_code == 400

    def test_connector_mode_tasks(self, rest_server):
        """Test task retrieval in connector mode."""
        rest_server.use_connector = True
        with rest_server.app.test_client() as client:
            response = client.get('/experimental/v1/tasks', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "fed-uuid"
            })
            assert response.status_code == 501

    def test_invalid_round_number(self, rest_server):
        """Test tensor retrieval with invalid round number."""
        with rest_server.app.test_client() as client:
            response = client.get('/experimental/v1/tensors/aggregated', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "fed-uuid",
                "tensor_name": "test_tensor",
                "round_number": "invalid"
            })
            assert response.status_code == 400
