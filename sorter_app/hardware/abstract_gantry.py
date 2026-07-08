"""Abstract base class for gantry communication.

This module defines the interface that all gantry clients must implement,
enabling dependency injection and testability.
"""

from abc import ABC, abstractmethod


class AbstractGantryClient(ABC):
    """Abstract interface for gantry hardware communication.

    This interface defines the contract for communicating with the ESP32
    gantry controller. Concrete implementations handle the actual UART
    communication (GantryClient) or simulate it (MockGantryClient).

    All methods are blocking with timeouts defined in the configuration.
    """

    @abstractmethod
    def connect(self) -> None:
        """Connect to the gantry hardware.

        Opens the serial port and waits for the ESP32 to send !READY
        notification (boot complete).

        Raises:
            GantryError: If connection fails or timeout expires.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the gantry hardware.

        Closes the serial port and releases resources.
        """
        pass

    @abstractmethod
    def home(self, axes: str = "XY") -> None:
        """Home the specified axes.

        Sends G28 command and waits for !HOMED notification.

        Args:
            axes: Axes to home. "X", "Y", or "XY" (default).

        Raises:
            GantryError: If homing fails or timeout expires.
        """
        pass

    @abstractmethod
    def move_to(self, x: float, y: float, feed_rate: int = 3000) -> None:
        """Move to absolute position.

        Sends G1 command and waits for !MOVE_DONE notification.

        Args:
            x: Target X position in millimeters.
            y: Target Y position in millimeters.
            feed_rate: Feed rate in mm/min (default: 3000 = 50 mm/s).

        Raises:
            GantryError: If move fails, out of bounds, or timeout expires.
        """
        pass

    @abstractmethod
    def open_gate(self) -> None:
        """Open the servo gate.

        Sends M3 command and waits for !GATE_DONE notification.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        pass

    @abstractmethod
    def close_gate(self) -> None:
        """Close the servo gate.

        Sends M5 command and waits for !GATE_DONE notification.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        pass

    @abstractmethod
    def get_position(self) -> tuple[float, float]:
        """Get current gantry position.

        Sends M114 command and parses the response.

        Returns:
            Tuple of (x_mm, y_mm) current position.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger emergency stop.

        Sends M112 command immediately. Does not wait for response.
        The gantry enters ESTOPPED state and requires reset() to recover.

        Raises:
            GantryError: If command fails to send.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset from emergency stop state.

        Sends M999 command and waits for !READY notification.

        Raises:
            GantryError: If reset fails or timeout expires.
        """
        pass

    @abstractmethod
    def wait_idle(self) -> None:
        """Wait for all queued moves to complete.

        Sends M400 command (sync barrier) and waits for response.

        Raises:
            GantryError: If command fails or timeout expires.
        """
        pass
