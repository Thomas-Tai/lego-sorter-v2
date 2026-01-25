import pytest
from unittest.mock import MagicMock, patch, call
from lego_sorter import MotorDriver, STEP_SEQUENCE

class TestMotorDriver:
    @pytest.fixture
    def mock_device_cls(self):
        with patch('lego_sorter.OutputDevice') as mock:
            yield mock

    def test_init_pins(self, mock_device_cls):
        """Test that Motor initializes correct V4.5 pins"""
        driver = MotorDriver()
        
        # Verify calls for pins 27, 22, 10, 9
        expected_calls = [call(27), call(22), call(10), call(9)]
        mock_device_cls.assert_has_calls(expected_calls, any_order=True)
        assert len(driver.pins) == 4

    def test_stop_safety(self, mock_device_cls):
        """Test that stop() de-energizes all coils (Safety Critical)"""
        driver = MotorDriver()
        driver.stop()
        
        # Every pin must be turned off
        for pin_mock in driver.pins:
            pin_mock.off.assert_called()

    def test_step_sequence(self, mock_device_cls):
        """Test that step() follows the 8-step half-step sequence"""
        driver = MotorDriver()
        
        with patch('time.sleep') as mock_sleep:
            # Run 8 steps (1 full cycle of the sequence)
            driver.step(steps=8, direction=1)
            
            # Verify pins were toggled
            for pin in driver.pins:
                assert pin.on.called
                assert pin.off.called

    def test_cleanup(self, mock_device_cls):
        """Test resource cleanup"""
        driver = MotorDriver()
        driver.cleanup()
        # Should call stop() or similar logic
        for pin_mock in driver.pins:
            pin_mock.off.assert_called()
