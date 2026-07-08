"""Tests for GantryClient and MockGantryClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from sorter_app.hardware import GantryClient, MockGantryClient, AbstractGantryClient
from sorter_app.hardware.gantry_client import ERROR_CODES
from sorter_app.domain.schemas import (
    GantryConfig,
    GantrySerialConfig,
    GantryMotionConfig,
    GantryServoConfig,
    GantrySimulationConfig,
)
from sorter_app.exceptions import GantryError


# Test fixtures
@pytest.fixture
def gantry_config() -> GantryConfig:
    """Create a test gantry configuration."""
    return GantryConfig(
        serial=GantrySerialConfig(
            port="/dev/ttyUSB0",
            baud_rate=115200,
            timeout_s=5.0,
            retry_count=2,
        ),
        motion=GantryMotionConfig(
            x_max_mm=355.0,
            y_max_mm=160.0,
            feed_rate_mm_min=3000,
            homing_required=True,
        ),
        servo=GantryServoConfig(
            open_angle_us=2500,
            close_angle_us=500,
            gate_hold_ms=300,
        ),
        simulation=GantrySimulationConfig(
            enabled=False,
            move_delay_s=0.1,
        ),
    )


class TestMockGantryClient:
    """Tests for MockGantryClient."""

    def test_mock_gantry_connect(self, gantry_config: GantryConfig) -> None:
        """Test connecting mock gantry."""
        client = MockGantryClient(gantry_config)
        assert not client._connected

        client.connect()
        assert client._connected

    def test_mock_gantry_disconnect(self, gantry_config: GantryConfig) -> None:
        """Test disconnecting mock gantry."""
        client = MockGantryClient(gantry_config)
        client.connect()
        assert client._connected

        client.disconnect()
        assert not client._connected

    def test_mock_gantry_home(self, gantry_config: GantryConfig) -> None:
        """Test homing mock gantry."""
        client = MockGantryClient(gantry_config)
        client.connect()
        client.home()

        assert client._homed
        assert client.get_position() == (0.0, 0.0)

    def test_mock_gantry_move_validates_bounds(
        self, gantry_config: GantryConfig
    ) -> None:
        """Test that move_to validates bounds."""
        client = MockGantryClient(gantry_config)
        client.connect()
        client.home()

        # Valid move
        client.move_to(100.0, 50.0)
        assert client.get_position() == (100.0, 50.0)

        # Out of bounds X
        with pytest.raises(GantryError, match="out of bounds"):
            client.move_to(400.0, 50.0)

        # Out of bounds Y
        with pytest.raises(GantryError, match="out of bounds"):
            client.move_to(100.0, 200.0)

    def test_mock_gantry_move_requires_home(self, gantry_config: GantryConfig) -> None:
        """Test that move_to requires homing first."""
        client = MockGantryClient(gantry_config)
        client.connect()

        with pytest.raises(GantryError, match="not homed"):
            client.move_to(100.0, 50.0)

    def test_mock_gantry_gate_operations(self, gantry_config: GantryConfig) -> None:
        """Test gate open/close operations."""
        client = MockGantryClient(gantry_config)
        client.connect()

        client.open_gate()
        assert client._gate_open

        client.close_gate()
        assert not client._gate_open

    def test_mock_gantry_emergency_stop(self, gantry_config: GantryConfig) -> None:
        """Test emergency stop and reset."""
        client = MockGantryClient(gantry_config)
        client.connect()
        client.home()

        client.emergency_stop()
        assert client._estopped

        # Operations should fail after e-stop
        with pytest.raises(GantryError, match="e-stop"):
            client.move_to(100.0, 50.0)

        # Reset from e-stop
        client.reset()
        assert not client._estopped

    def test_mock_gantry_position_tracking(self, gantry_config: GantryConfig) -> None:
        """Test position tracking."""
        client = MockGantryClient(gantry_config)
        client.connect()
        client.home()

        assert client.get_position() == (0.0, 0.0)

        client.move_to(150.0, 75.0)
        assert client.get_position() == (150.0, 75.0)

        client.move_to(0.0, 0.0)
        assert client.get_position() == (0.0, 0.0)


class TestGantryClientInterface:
    """Tests for GantryClient interface (mock serial tests)."""

    def test_send_command_format(self, gantry_config: GantryConfig) -> None:
        """Test that send_command formats commands correctly."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            # Re-import to get mocked serial
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Set up mock to return !READY for connect, then ok for commands
            mock_port.readline.side_effect = [b"!READY\n", b"ok\n"]
            client.connect()

            # Reset the mock to check write call
            mock_port.write.reset_mock()
            mock_port.flush.reset_mock()
            mock_port.readline.return_value = b"ok\n"

            # Send a command via the private method
            client._send_command("G28")

            # Verify the command format
            mock_port.write.assert_called_once()
            written = mock_port.write.call_args[0][0]
            assert written == b"G28\n"

    def test_parse_ok_response(self, gantry_config: GantryConfig) -> None:
        """Test parsing ok response."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Connect with !READY, then return ok for commands
            mock_port.readline.side_effect = [b"!READY\n", b"ok\n"]
            client.connect()

            response = client._send_command("G28")
            assert response == "ok"

    def test_parse_position_response(self, gantry_config: GantryConfig) -> None:
        """Test parsing M114 position response."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Connect with !READY, then return position for M114
            mock_port.readline.side_effect = [
                b"!READY\n",
                b"ok X:123.5 Y:456.7\n",
            ]
            client.connect()

            x, y = client.get_position()
            assert x == 123.5
            assert y == 456.7

    def test_error_response_raises_exception(self, gantry_config: GantryConfig) -> None:
        """Test that error response raises GantryError."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Connect with !READY, then return error for command
            # Need enough responses for retry attempts
            mock_port.readline.side_effect = [
                b"!READY\n",
                b"error:4\n",  # First attempt - error raised
                b"",  # Second attempt - timeout
                b"",  # Third attempt - timeout
            ]
            client.connect()

            with pytest.raises(GantryError, match="failed after 3 attempts"):
                client._send_command("G1 X100 Y100")

    def test_bounds_validation(self, gantry_config: GantryConfig) -> None:
        """Test that move_to validates bounds."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Connect with !READY
            mock_port.readline.return_value = b"!READY\n"
            client.connect()

            # Out of bounds X
            with pytest.raises(GantryError, match="out of bounds"):
                client.move_to(400.0, 50.0)

            # Out of bounds Y
            with pytest.raises(GantryError, match="out of bounds"):
                client.move_to(100.0, 200.0)

    def test_timeout_retry(self, gantry_config: GantryConfig) -> None:
        """Test that timeout triggers retry."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Connect: !READY, then timeout + ok for command
            mock_port.readline.side_effect = [
                b"!READY\n",
                b"",  # timeout
                b"ok\n",  # retry success
            ]
            client.connect()

            # Should succeed after retry
            response = client._send_command("G28")
            assert response == "ok"

    def test_notification_skipped_in_response(
        self, gantry_config: GantryConfig
    ) -> None:
        """Test that notifications are skipped when reading response."""
        mock_serial_module = MagicMock()
        mock_port = MagicMock()
        mock_serial_module.Serial.return_value = mock_port

        with patch.dict("sys.modules", {"serial": mock_serial_module}):
            from sorter_app.hardware.gantry_client import (
                GantryClient as MockedGantryClient,
            )

            client = MockedGantryClient(gantry_config)

            # Connect, then notification skipped, ok returned
            mock_port.readline.side_effect = [
                b"!READY\n",
                b"!MOVE_DONE\n",
                b"ok\n",
            ]
            client.connect()

            response = client._send_command("G28")
            assert response == "ok"


class TestAbstractGantryClient:
    """Tests for AbstractGantryClient interface."""

    def test_abstract_methods_exist(self) -> None:
        """Test that all required abstract methods are defined."""
        from abc import ABC

        # Check that AbstractGantryClient is an ABC
        assert issubclass(AbstractGantryClient, ABC)

        # Check that key methods are abstract (by checking __isabstractmethod__)
        expected_methods = [
            "connect",
            "disconnect",
            "home",
            "move_to",
            "open_gate",
            "close_gate",
            "get_position",
            "emergency_stop",
            "reset",
            "wait_idle",
        ]

        for method_name in expected_methods:
            method = getattr(AbstractGantryClient, method_name, None)
            assert method is not None, f"Missing method: {method_name}"
            assert getattr(
                method, "__isabstractmethod__", False
            ), f"Method {method_name} is not abstract"

    def test_mock_implements_abstract(self, gantry_config: GantryConfig) -> None:
        """Test that MockGantryClient implements all abstract methods."""
        client = MockGantryClient(gantry_config)

        # Verify it's an instance of AbstractGantryClient
        assert isinstance(client, AbstractGantryClient)

        # Verify all methods exist
        assert hasattr(client, "connect")
        assert hasattr(client, "disconnect")
        assert hasattr(client, "home")
        assert hasattr(client, "move_to")
        assert hasattr(client, "open_gate")
        assert hasattr(client, "close_gate")
        assert hasattr(client, "get_position")
        assert hasattr(client, "emergency_stop")
        assert hasattr(client, "reset")
        assert hasattr(client, "wait_idle")
