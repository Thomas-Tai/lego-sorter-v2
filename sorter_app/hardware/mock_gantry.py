"""MockGantryClient - Simulated gantry for testing without hardware.

This module provides a mock implementation of AbstractGantryClient that
simulates gantry behavior without requiring physical hardware.
"""

import logging
import time
from typing import Optional

from ..exceptions import GantryError
from ..domain.schemas import GantryConfig
from .abstract_gantry import AbstractGantryClient

logger = logging.getLogger(__name__)


class MockGantryClient(AbstractGantryClient):
    """Mock implementation of AbstractGantryClient for simulation.

    This client simulates gantry behavior without real hardware.
    It tracks position, validates bounds, and simulates delays.

    Attributes:
        config: GantryConfig with motion and simulation settings.
    """

    def __init__(self, config: GantryConfig) -> None:
        """Initialize MockGantryClient with configuration.

        Args:
            config: Validated GantryConfig from gantry.yaml.
        """
        self._config = config
        self._connected = False
        self._homed = False
        self._position: tuple[float, float] = (0.0, 0.0)
        self._gate_open = False
        self._estopped = False
        logger.info("MockGantryClient initialized (simulation mode)")

    def connect(self) -> None:
        """Simulate connecting to the gantry hardware."""
        if self._connected:
            logger.warning("MockGantryClient already connected")
            return

        # Simulate boot delay
        time.sleep(0.1)
        self._connected = True
        self._estopped = False
        logger.info("MockGantryClient connected (simulated !READY)")

    def disconnect(self) -> None:
        """Simulate disconnecting from the gantry hardware."""
        self._connected = False
        self._homed = False
        logger.info("MockGantryClient disconnected")

    def home(self, axes: str = "XY") -> None:
        """Simulate homing the specified axes.

        Args:
            axes: Axes to home. "X", "Y", or "XY" (default).

        Raises:
            GantryError: If not connected or e-stopped.
        """
        self._check_connected()

        # Simulate homing delay
        time.sleep(self._config.simulation.move_delay_s)

        self._position = (0.0, 0.0)
        self._homed = True
        logger.info("MockGantryClient homed: %s (position reset to 0,0)", axes)

    def move_to(self, x: float, y: float, feed_rate: int = 3000) -> None:
        """Simulate moving to absolute position.

        Args:
            x: Target X position in millimeters.
            y: Target Y position in millimeters.
            feed_rate: Feed rate in mm/min (default: 3000).

        Raises:
            GantryError: If out of bounds, not homed, or not connected.
        """
        self._check_connected()
        self._check_homed()
        self._check_estopped()

        # Validate bounds
        if x < 0 or x > self._config.motion.x_max_mm:
            raise GantryError(
                f"X position {x} out of bounds [0, {self._config.motion.x_max_mm}]"
            )
        if y < 0 or y > self._config.motion.y_max_mm:
            raise GantryError(
                f"Y position {y} out of bounds [0, {self._config.motion.y_max_mm}]"
            )

        # Simulate move delay
        time.sleep(self._config.simulation.move_delay_s)

        self._position = (x, y)
        logger.debug("MockGantryClient moved to (%.1f, %.1f)", x, y)

    def open_gate(self) -> None:
        """Simulate opening the servo gate."""
        self._check_connected()
        self._check_estopped()

        time.sleep(self._config.simulation.move_delay_s)
        self._gate_open = True
        logger.debug("MockGantryClient gate opened")

    def close_gate(self) -> None:
        """Simulate closing the servo gate."""
        self._check_connected()
        self._check_estopped()

        time.sleep(self._config.simulation.move_delay_s)
        self._gate_open = False
        logger.debug("MockGantryClient gate closed")

    def get_position(self) -> tuple[float, float]:
        """Get simulated gantry position.

        Returns:
            Tuple of (x_mm, y_mm) current position.
        """
        self._check_connected()
        return self._position

    def emergency_stop(self) -> None:
        """Simulate triggering emergency stop."""
        self._estopped = True
        logger.warning("MockGantryClient emergency stop triggered")

    def reset(self) -> None:
        """Simulate resetting from emergency stop state."""
        self._check_connected()

        time.sleep(self._config.simulation.move_delay_s)
        self._estopped = False
        self._homed = False
        logger.info("MockGantryClient reset from e-stop")

    def wait_idle(self) -> None:
        """Simulate waiting for all queued moves to complete."""
        self._check_connected()
        # In simulation, moves are immediate, so this is a no-op
        logger.debug("MockGantryClient idle")

    def _check_connected(self) -> None:
        """Check that the client is connected.

        Raises:
            GantryError: If not connected.
        """
        if not self._connected:
            raise GantryError("MockGantryClient not connected")

    def _check_homed(self) -> None:
        """Check that the gantry has been homed.

        Raises:
            GantryError: If not homed.
        """
        if not self._homed:
            raise GantryError("MockGantryClient not homed - call home() first")

    def _check_estopped(self) -> None:
        """Check that the gantry is not in e-stop state.

        Raises:
            GantryError: If e-stopped.
        """
        if self._estopped:
            raise GantryError("MockGantryClient in e-stop state - call reset() first")
