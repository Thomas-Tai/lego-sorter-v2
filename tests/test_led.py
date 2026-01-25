import pytest
from unittest.mock import MagicMock, patch
import time

# Mock gpiozero for Windows environment
import sys

sys.modules["gpiozero"] = MagicMock()
from gpiozero import PWMLED

from lego_sorter import LedDriver


class TestLedDriver:
    @pytest.fixture
    def mock_led_cls(self):
        with patch("lego_sorter.PWMLED") as mock:
            yield mock

    def test_init(self, mock_led_cls):
        """Test that LED is initialized with correct pin"""
        driver = LedDriver(pin=23)
        mock_led_cls.assert_called_with(23, frequency=100)
        assert driver.led == mock_led_cls.return_value

    def test_on_off(self, mock_led_cls):
        """Test simple on/off control"""
        driver = LedDriver(pin=23)
        driver.on()
        driver.led.on.assert_called_once()

        driver.off()
        driver.led.off.assert_called_once()

    def test_fade_in(self, mock_led_cls):
        """Test fade-in sequencing"""
        driver = LedDriver(pin=23)

        # Mock time.sleep to run fast
        with patch("time.sleep") as mock_sleep:
            driver.fade_in(duration=0.5)

            # verify duty cycle changes
            assert driver.led.value == 1.0  # Should end at 1.0
            assert mock_sleep.call_count > 0

    def test_fade_out(self, mock_led_cls):
        """Test fade-out sequencing"""
        driver = LedDriver(pin=23)

        with patch("time.sleep") as mock_sleep:
            driver.fade_out(duration=0.5)

            assert driver.led.value == 0.0  # Should end at 0.0
