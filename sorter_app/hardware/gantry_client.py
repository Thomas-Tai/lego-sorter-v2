"""GantryClient - UART communication with ESP32 gantry controller.

This module provides the GantryClient class which implements the
AbstractGantryClient interface for real hardware communication via UART.
"""

import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Optional, Any

from ..exceptions import GantryError, GantryProtocolError, GantryTimeoutError
from ..domain.schemas import GantryConfig
from .abstract_gantry import AbstractGantryClient

if TYPE_CHECKING:
    import serial

logger = logging.getLogger(__name__)

# Error code mapping from SM-DES-003
ERROR_CODES: dict[int, str] = {
    1: "Unknown command",
    2: "Invalid parameter",
    3: "Move out of bounds",
    4: "Not homed",
    5: "Emergency stop active",
    6: "Busy",
}


class GantryClient(AbstractGantryClient):
    """Concrete implementation of AbstractGantryClient using UART.

    This client communicates with the ESP32 firmware via serial port,
    sending G-code commands and parsing responses/notifications.

    Thread Safety:
        All serial operations are protected by a lock. Multiple threads
        can safely call methods, though commands are processed sequentially.

    Attributes:
        config: GantryConfig with serial, motion, and servo settings.
    """

    def __init__(self, config: GantryConfig) -> None:
        """Initialize GantryClient with configuration.

        Args:
            config: Validated GantryConfig from gantry.yaml.
        """
        self._config = config
        self._serial: Optional["serial.Serial"] = None
        self._lock = threading.Lock()
        self._connected = False

        # Import serial here to allow module import without hardware
        try:
            import serial as _serial

            self._serial_module = _serial
        except ImportError as e:
            logger.warning(
                "pyserial not installed - GantryClient will not work. "
                "Install with: pip install pyserial"
            )
            raise ImportError(
                "pyserial required for GantryClient. Install with: pip install pyserial"
            ) from e

    def connect(self) -> None:
        """Connect to the gantry hardware.

        Opens the serial port and waits for !READY notification.

        Raises:
            GantryError: If connection fails or timeout expires.
        """
        with self._lock:
            if self._connected:
                logger.warning("GantryClient already connected")
                return

            try:
                self._serial = self._serial_module.Serial(
                    port=self._config.serial.port,
                    baudrate=self._config.serial.baud_rate,
                    timeout=self._config.serial.timeout_s,
                )
                self._connected = True
                logger.info(
                    "Opened serial port %s at %d baud",
                    self._config.serial.port,
                    self._config.serial.baud_rate,
                )
            except Exception as e:
                raise GantryError(
                    f"Failed to open serial port {self._config.serial.port}: {e}"
                ) from e

        # Wait for !READY notification (ESP32 boot complete)
        self._wait_for_notification("!READY", timeout=10.0)
        logger.info("Gantry connected - ESP32 ready")

    def disconnect(self) -> None:
        """Disconnect from the gantry hardware."""
        with self._lock:
            if self._serial is not None and self._connected:
                try:
                    self._serial.close()
                except Exception as e:
                    logger.warning("Error closing serial port: %s", e)
                finally:
                    self._serial = None
                    self._connected = False
                    logger.info("Gantry disconnected")

    def home(self, axes: str = "XY") -> None:
        """Home the specified axes.

        Args:
            axes: Axes to home. "X", "Y", or "XY" (default).

        Raises:
            GantryError: If homing fails or timeout expires.
        """
        if axes == "XY":
            cmd = "G28"
        elif axes in ("X", "Y"):
            cmd = f"G28 {axes}"
        else:
            raise GantryError(f"Invalid axes for homing: {axes}")

        self._send_command(cmd)
        self._wait_for_notification("!HOMED", timeout=30.0)
        logger.info("Gantry homed: %s", axes)

    def move_to(self, x: float, y: float, feed_rate: int = 3000) -> None:
        """Move to absolute position.

        Args:
            x: Target X position in millimeters.
            y: Target Y position in millimeters.
            feed_rate: Feed rate in mm/min (default: 3000 = 50 mm/s).

        Raises:
            GantryError: If move fails, out of bounds, or timeout expires.
        """
        # Validate bounds
        if x < 0 or x > self._config.motion.x_max_mm:
            raise GantryError(
                f"X position {x} out of bounds [0, {self._config.motion.x_max_mm}]"
            )
        if y < 0 or y > self._config.motion.y_max_mm:
            raise GantryError(
                f"Y position {y} out of bounds [0, {self._config.motion.y_max_mm}]"
            )

        cmd = f"G1 X{x:.1f} Y{y:.1f} F{feed_rate}"
        self._send_command(cmd)
        self._wait_for_notification("!MOVE_DONE", timeout=30.0)
        logger.debug("Gantry moved to (%.1f, %.1f)", x, y)

    def open_gate(self) -> None:
        """Open the servo gate.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        self._send_command("M3")
        self._wait_for_notification("!GATE_DONE", timeout=5.0)
        logger.debug("Gate opened")

    def close_gate(self) -> None:
        """Close the servo gate.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        self._send_command("M5")
        self._wait_for_notification("!GATE_DONE", timeout=5.0)
        logger.debug("Gate closed")

    def get_position(self) -> tuple[float, float]:
        """Get current gantry position.

        Returns:
            Tuple of (x_mm, y_mm) current position.

        Raises:
            GantryTimeoutError: If the command times out.
            GantryProtocolError: If the response cannot be parsed.
            GantryError: If command fails.
        """
        response = self._send_command("M114")
        # Parse "ok X:123.0 Y:456.0" format
        match = re.search(r"X:([0-9.]+)\s+Y:([0-9.]+)", response)
        if not match:
            raise GantryProtocolError(f"Failed to parse position response: {response}")

        x = float(match.group(1))
        y = float(match.group(2))
        return (x, y)

    def emergency_stop(self) -> None:
        """Trigger emergency stop.

        Sends M112 command immediately. Does not wait for response.
        """
        with self._lock:
            if not self._connected or self._serial is None:
                raise GantryError("Cannot send e-stop: not connected")

            try:
                self._serial.write(b"M112\n")
                self._serial.flush()
                logger.warning("Emergency stop sent")
            except Exception as e:
                raise GantryError(f"Failed to send e-stop: {e}") from e

    def reset(self) -> None:
        """Reset from emergency stop state.

        Raises:
            GantryError: If reset fails or timeout expires.
        """
        self._send_command("M999")
        self._wait_for_notification("!READY", timeout=10.0)
        logger.info("Gantry reset from e-stop")

    def wait_idle(self) -> None:
        """Wait for all queued moves to complete.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        self._send_command("M400")
        logger.debug("Gantry idle")

    def _send_command(self, cmd: str) -> str:
        """Send a command and wait for response.

        Args:
            cmd: G-code command string (without newline).

        Returns:
            Response string (e.g., "ok" or "ok X:123 Y:456").

        Raises:
            GantryError: If error response received or timeout expires.
        """
        with self._lock:
            if not self._connected or self._serial is None:
                raise GantryError(f"Cannot send command '{cmd}': not connected")

            retries = self._config.serial.retry_count
            last_error: Optional[Exception] = None

            for attempt in range(retries + 1):
                try:
                    # Send command
                    full_cmd = f"{cmd}\n"
                    self._serial.write(full_cmd.encode("ascii"))
                    self._serial.flush()
                    logger.debug("Sent: %s", cmd)

                    # Read response
                    response = self._read_response()
                    return response

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Command '%s' failed (attempt %d/%d): %s",
                        cmd,
                        attempt + 1,
                        retries + 1,
                        e,
                    )
                    continue

            raise GantryError(
                f"Command '{cmd}' failed after {retries + 1} attempts: {last_error}"
            ) from last_error

    def _read_response(self) -> str:
        """Read response from serial port.

        Reads lines until a response (not notification) is received.

        Returns:
            Response string.

        Raises:
            GantryTimeoutError: If no response is received before the
                serial port timeout elapses.
            GantryProtocolError: If the response is malformed or an
                unrecognized token (e.g. an unparseable NACK line or
                an unexpected response format).
            GantryError: If a well-formed firmware error (NACK) is
                received.
        """
        assert self._serial is not None

        while True:
            line_bytes = self._serial.readline()
            if not line_bytes:
                raise GantryTimeoutError("Serial read timeout")

            line = line_bytes.decode("ascii", errors="replace").strip()
            logger.debug("Received: %s", line)

            # Skip empty lines
            if not line:
                continue

            # Notifications start with "!" - skip them (they're handled elsewhere)
            if line.startswith("!"):
                # Could queue for notification handling, but for now just log
                logger.debug("Notification: %s", line)
                continue

            # Check for error response
            if line.startswith("error:"):
                try:
                    code = int(line.split(":")[1].split()[0])
                    msg = ERROR_CODES.get(code, "Unknown error")
                    raise GantryError(f"Firmware error {code}: {msg}")
                except (IndexError, ValueError):
                    raise GantryProtocolError(f"Firmware error: {line}")

            # Check for ok response
            if line.startswith("ok"):
                return line

            # Unknown response format
            logger.warning("Unexpected response format: %s", line)
            raise GantryProtocolError(f"Unexpected response: {line}")

    def _wait_for_notification(self, expected: str, timeout: float) -> None:
        """Wait for a specific notification.

        Args:
            expected: Expected notification string (e.g., "!READY").
            timeout: Timeout in seconds.

        Raises:
            GantryTimeoutError: If timeout expires.
            GantryError: If an emergency stop notification is received
                while waiting.
        """
        start_time = time.monotonic()
        assert self._serial is not None

        # Temporarily adjust serial timeout for polling
        original_timeout = self._serial.timeout
        self._serial.timeout = 0.1  # Polling interval

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    raise GantryTimeoutError(
                        f"Timeout waiting for notification '{expected}'"
                    )

                with self._lock:
                    line_bytes = self._serial.readline()
                    if line_bytes:
                        line = line_bytes.decode("ascii", errors="replace").strip()
                        logger.debug("Notification check: %s", line)

                        if line == expected:
                            return

                        # Handle unexpected !ESTOP notification
                        if line == "!ESTOP":
                            raise GantryError("Emergency stop triggered")

        finally:
            with self._lock:
                self._serial.timeout = original_timeout
