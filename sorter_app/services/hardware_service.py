"""Concrete hardware service for Raspberry Pi GPIO control."""

import logging

from sorter_app.exceptions import HardwareError
from modules.hardware.led import LedDriver
from modules.hardware.motor import MotorDriver

from .base_service import AbstractHardwareService

logger = logging.getLogger(__name__)


class RaspberryPiHardwareService(AbstractHardwareService):
    """Hardware service that delegates to MotorDriver and LedDriver.

    Bridges the abstract service interface to concrete GPIO drivers
    in modules/hardware/.
    """

    def __init__(self) -> None:
        self._motor = MotorDriver()
        self._led = LedDriver()

    def setup(self) -> None:
        """Initialize hardware drivers."""
        logger.info("Hardware service initialized")

    def turn_turntable(self, degrees: int) -> None:
        """Rotate turntable by given degrees.

        Args:
            degrees: Rotation angle. Positive = clockwise.

        Raises:
            HardwareError: If motor fails to step.
        """
        try:
            steps = int(abs(degrees) * 4096 / 360)
            direction = 1 if degrees >= 0 else -1
            self._motor.step(steps, direction)
        except Exception as e:
            raise HardwareError(f"Motor step failed: {e}") from e

    def set_led_power(self, is_on: bool) -> None:
        """Turn LED ring on or off.

        Args:
            is_on: True to enable, False to disable.
        """
        if is_on:
            self._led.on()
        else:
            self._led.off()

    def cleanup(self) -> None:
        """Release hardware resources."""
        self._motor.cleanup()
        self._led.cleanup()
