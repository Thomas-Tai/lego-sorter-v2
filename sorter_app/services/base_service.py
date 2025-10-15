# In sorter_app/services/base_service.py

from abc import ABC, abstractmethod


class AbstractHardwareService(ABC):
    """
    An abstract base class that defines the interface for all hardware services.

    This defines the "contract" that any hardware implementation (real or mock)
    must adhere to. High-level logic will depend on this abstraction, not on
    a concrete implementation.
    """

    @abstractmethod
    def setup(self) -> None:
        """Initializes hardware pins and settings."""
        pass

    @abstractmethod
    def turn_turntable(self, degrees: int) -> None:
        """
        Turns the data acquisition turntable by a specified number of degrees.

        Args:
            degrees (int): The number of degrees to turn. Can be positive
                           (clockwise) or negative (counter-clockwise).
        """
        pass

    @abstractmethod
    def set_led_power(self, is_on: bool) -> None:
        """
        Turns the data acquisition lighting on or off.

        Args:
            is_on (bool): True to turn the LEDs on, False to turn them off.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up hardware resources (e.g., GPIO pins)."""
        pass
