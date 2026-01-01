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


class AbstractVisionService(ABC):
    """Abstract base class defining the interface for vision/camera services.

    This defines the contract that any vision implementation (real or mock)
    must adhere to. High-level logic will depend on this abstraction.
    """

    @abstractmethod
    def capture_image(self, filepath: str) -> bool:
        """Captures an image and saves it to the given path.

        Args:
            filepath: Absolute path where the image should be saved.

        Returns:
            True if the image was captured and saved successfully,
            False otherwise.

        Raises:
            CameraError: If the camera is not available or fails to capture.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases camera resources.

        Should be called when the vision service is no longer needed
        to free up the camera device for other processes.
        """
        pass

