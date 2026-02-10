import pytest
from unittest.mock import MagicMock, patch, call
from sorter_app.controllers.sorter_controller import LegoSorter
from sorter_app.state_machine import SorterState


class TestLegoSorter:
    @pytest.fixture
    def sorter(self):
        # Patch hardware drivers to prevent real hardware access during init
        with patch(
            "sorter_app.controllers.sorter_controller.Button"
        ) as mock_btn, patch(
            "sorter_app.controllers.sorter_controller.LedDriver"
        ) as mock_led, patch(
            "sorter_app.controllers.sorter_controller.MotorDriver"
        ) as mock_motor:

            sorter = LegoSorter()
            assert isinstance(sorter.led, MagicMock), "LedDriver was not mocked!"
            yield sorter

    def test_initial_state(self, sorter):
        """Test initial state is IDLE"""
        assert sorter.state == SorterState.IDLE
        # Components should be initialized
        assert sorter.led is not None
        assert sorter.motor is not None
        assert sorter.button is not None

    def test_workflow_cycle(self, sorter):
        """Test the full workflow: IDLE -> RUNNING -> IDLE"""

        with patch("time.sleep"):  # skip delays
            sorter.start_cycle()

            # Assertions using call_count for robustness
            assert (
                sorter.led.fade_in.call_count == 1
            ), "Expected fade_in to be called once"

            # SORT: Motor Action
            assert (
                sorter.motor.run_for.call_count == 1
            ), "Expected motor.run_for to be called once"

            # DONE: LED Fade Out & Cleanup
            assert (
                sorter.led.fade_out.call_count == 1
            ), "Expected fade_out to be called once"

            # Return to IDLE
            assert sorter.state == SorterState.IDLE

    def test_button_trigger_handler(self, sorter):
        """Test that update() checks button properly"""
        sorter.button.is_pressed = True

        # Mock start_cycle so we don't run the whole thing
        sorter.start_cycle = MagicMock()

        sorter.update()  # Single loop iteration
        sorter.start_cycle.assert_called_once()

    def test_cleanup(self, sorter):
        """Test cleanup calls drivers"""
        sorter.cleanup()
        sorter.led.cleanup.assert_called()
        sorter.motor.cleanup.assert_called()
