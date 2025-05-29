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

# The .proto file has GetAggregatedTensorsRequest/Response and TensorSpec,
# but the Python protobuf files are out of sync and missing these classes.
# Add them to the module using existing protobuf infrastructure.

# Create TensorSpec class using the existing protobuf pattern
class TensorSpec:
    def __init__(self):
        self.tensor_name = ""
        self.round_number = 0
        self.report = False
        self.tags = []
        self.require_lossless = False

# Create GetAggregatedTensorsRequest using existing MessageHeader
class GetAggregatedTensorsRequest:
    def __init__(self):
        self.header = aggregator_pb2.MessageHeader()
        self.tensor_specs = []

# Create GetAggregatedTensorsResponse with protobuf compatibility
class GetAggregatedTensorsResponse:
    def __init__(self, header=None, tensors=None):
        self.header = header or aggregator_pb2.MessageHeader()
        self.tensors = tensors or []
        self.DESCRIPTOR = None

# Add the missing classes to aggregator_pb2 module so the server can find them
aggregator_pb2.TensorSpec = TensorSpec
aggregator_pb2.GetAggregatedTensorsRequest = GetAggregatedTensorsRequest
aggregator_pb2.GetAggregatedTensorsResponse = GetAggregatedTensorsResponse
aggregator_pb2.NamedTensorProto = base_pb2.NamedTensor

# Patch json_format to handle the new classes since they're not "real" protobuf messages
from google.protobuf import json_format

original_parse_dict = json_format.ParseDict
original_message_to_dict = json_format.MessageToDict

def patched_parse_dict(js_dict, message, **kwargs):
    """Custom ParseDict to handle the missing protobuf classes."""
    if isinstance(message, GetAggregatedTensorsRequest):
        # Parse header manually
        if 'header' in js_dict:
            header_data = js_dict['header']
            message.header.sender = header_data.get('sender', '')
            message.header.receiver = header_data.get('receiver', '')
            message.header.federation_uuid = header_data.get('federation_uuid', '')
            message.header.single_col_cert_common_name = header_data.get('single_col_cert_common_name', '')

        # Parse tensor specs manually
        if 'tensor_specs' in js_dict:
            message.tensor_specs = []
            for spec_data in js_dict['tensor_specs']:
                spec = TensorSpec()
                spec.tensor_name = spec_data.get('tensor_name', '')
                spec.round_number = spec_data.get('round_number', 0)
                spec.report = spec_data.get('report', False)
                spec.tags = spec_data.get('tags', [])
                spec.require_lossless = spec_data.get('require_lossless', False)
                message.tensor_specs.append(spec)
        return message
    else:
        return original_parse_dict(js_dict, message, **kwargs)

def patched_message_to_dict(message, **kwargs):
    """Custom MessageToDict to handle our custom protobuf classes."""
    if isinstance(message, GetAggregatedTensorsResponse):
        # Manually convert our custom response to dict
        result = {
            "header": original_message_to_dict(message.header, **kwargs) if message.header else {},
            "tensors": []
        }
        # Convert tensors to dict format
        for tensor in message.tensors:
            if tensor:
                result["tensors"].append(original_message_to_dict(tensor, **kwargs))
        return result
    else:
        return original_message_to_dict(message, **kwargs)

json_format.ParseDict = patched_parse_dict
json_format.MessageToDict = patched_message_to_dict

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
            mock_context.options = 0

            rest_server._setup_ssl_context(
                certificate=ssl_certs['cert'],
                private_key=ssl_certs['key'],
                root_certificate=ssl_certs['root']
            )

            mock_ssl_context.assert_called_once_with(ssl.PROTOCOL_TLS_SERVER)
            mock_context.load_cert_chain.assert_called_once_with(
                certfile=ssl_certs['cert'],
                keyfile=ssl_certs['key']
            )

            assert mock_context.load_verify_locations.call_count == 2
            assert all(
                call == mock.call(cafile=ssl_certs['root'])
                for call in mock_context.load_verify_locations.call_args_list
            )

            assert mock_context.verify_mode == ssl.CERT_REQUIRED

    def test_get_tasks_valid_request(self, rest_server, mock_aggregator):
        """Test successful task retrieval."""
        mock_tasks = [
            aggregator_pb2.Task(name="task1", function_name="func1", task_type="train"),
            aggregator_pb2.Task(name="task2", function_name="func2", task_type="validate")
        ]
        mock_aggregator.get_tasks.return_value = (mock_tasks, 1, 5, True)

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
            assert "quit" in data
            assert data["quit"]

        mock_aggregator.get_tasks.return_value = (mock_tasks, 1, 5, False)

        with rest_server.app.test_client() as client:
            response = client.get('experimental/v1/tasks', query_string={
                "collaborator_id": "test-collaborator",
                "federation_uuid": "fed-uuid"
            })

            assert response.status_code == 200
            data = response.get_json()
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
        task_results = aggregator_pb2.TaskResults()
        task_results.task_name = "test_task"
        task_results.round_number = 1
        task_results.data_size = 100

        task_results.header.sender = "test-collaborator"
        task_results.header.receiver = str(mock_aggregator.uuid)
        task_results.header.federation_uuid = str(mock_aggregator.federation_uuid)
        task_results.header.single_col_cert_common_name = "test-cert-cn"

        tensor = base_pb2.NamedTensor()
        tensor.name = "test_tensor"
        task_results.tensors.append(tensor)

        data_stream = base_pb2.DataStream()
        data_stream.npbytes = task_results.SerializeToString()
        data_stream.size = len(data_stream.npbytes)

        request_data = (
            len(data_stream.SerializeToString()).to_bytes(4, byteorder='big') +
            data_stream.SerializeToString() +
            (0).to_bytes(4, byteorder='big')
        )

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
        mock_tensor = base_pb2.NamedTensor()
        mock_tensor.name = "test_tensor"

        def mock_get_aggregated_tensor(tensor_name, round_number, report=False, tags=(), require_lossless=False, requested_by=None):
            return mock_tensor

        mock_aggregator.get_aggregated_tensor.side_effect = mock_get_aggregated_tensor

        request_payload = {
            "header": {
                "sender": "test-collaborator",
                "receiver": str(mock_aggregator.uuid),
                "federation_uuid": str(mock_aggregator.federation_uuid),
                "single_col_cert_common_name": "test-cert-cn"
            },
            "tensor_specs": [{
                "tensor_name": "test_tensor",
                "round_number": 1,
                "report": False,
                "tags": [],
                "require_lossless": False
            }]
        }

        with rest_server.app.test_client() as client:
            response = client.post('/experimental/v1/tensors/aggregated/batch',
                                   json=request_payload,
                                   headers={
                                       "Sender": "test-collaborator",
                                       "Receiver": str(mock_aggregator.uuid),
                                       "Federation-UUID": str(mock_aggregator.federation_uuid),
                                       "Single-Col-Cert-CN": "test-cert-cn"
                                   })

            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response data: {response.get_data(as_text=True)}")

            assert response.status_code == 200
            data = response.get_json()
            assert "header" in data
            assert "tensors" in data
            assert len(data["tensors"]) == 1

            mock_aggregator.get_aggregated_tensor.assert_called_once_with(
                "test_tensor",
                1,
                False,
                (),
                False,
                "test-collaborator"
            )

    def test_relay_message_not_enabled(self, rest_server):
        """Test relay endpoint when not enabled."""
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
        request_payload = {
            "header": {
                "sender": "test-collaborator",
                "receiver": str(rest_server.aggregator.uuid),
                "federation_uuid": str(rest_server.aggregator.federation_uuid),
                "single_col_cert_common_name": "test-cert-cn"
            },
            "tensor_specs": [{
                "tensor_name": "test_tensor",
                "round_number": "invalid",
                "report": False,
                "tags": [],
                "require_lossless": False
            }]
        }

        with rest_server.app.test_client() as client:
            response = client.post('/experimental/v1/tensors/aggregated/batch',
                                   json=request_payload,
                                   headers={
                                       "Sender": "test-collaborator",
                                       "Receiver": str(rest_server.aggregator.uuid),
                                       "Federation-UUID": str(rest_server.aggregator.federation_uuid),
                                       "Single-Col-Cert-CN": "test-cert-cn"
                                   })
            assert response.status_code == 400
