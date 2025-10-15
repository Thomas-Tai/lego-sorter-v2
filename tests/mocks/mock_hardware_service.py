# In tests/mocks/mock_hardware_service.py

from sorter_app.services.base_service import AbstractHardwareService


class MockHardwareService(AbstractHardwareService):
    """
    A mock implementation of the Hardware Service for testing purposes.

    This class does not interact with any real hardware. Instead, it records
    the calls made to its methods, allowing tests to assert that high-level
    logic is correctly interacting with the hardware layer.
    """

    def __init__(self):
        self.setup_called = False
        self.cleanup_called = False
        self.led_is_on = None
        self.turntable_turned_by = 0

    def setup(self) -> None:
        self.setup_called = True

    def turn_turntable(self, degrees: int) -> None:
        self.turntable_turned_by += degrees

    def set_led_power(self, is_on: bool) -> None:
        self.led_is_on = is_on

    def cleanup(self) -> None:
        self.cleanup_called = True
