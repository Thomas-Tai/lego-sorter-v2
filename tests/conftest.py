import pytest
import sys
from unittest.mock import MagicMock

# Mock gpiozero and rpi-lgpio before any test imports modules that use them
# This forces "lego_sorter.py" to use the Mocks instead of real/installed packages
# preventing hardware access errors on Windows/CI.

module_mock = MagicMock()
sys.modules['gpiozero'] = module_mock
sys.modules['gpiozero.pins.pigpio'] = module_mock
sys.modules['rpi_lgpio'] = module_mock

@pytest.fixture(autouse=True)
def mock_gpiozero():
    """Ensure gpiozero is mocked for every test."""
    yield module_mock
