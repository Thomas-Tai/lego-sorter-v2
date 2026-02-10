"""Sorter controller - main orchestrator for the Lego sorting cycle.

Manages the state machine transitions and coordinates hardware drivers
to execute sorting cycles triggered by button press.
"""

import logging
import time

try:
    from gpiozero import Button
except ImportError:

    class Button:
        """Fallback for non-Pi environments."""

        def __init__(self, *args, **kwargs):
            pass

        @property
        def is_pressed(self) -> bool:
            return False


from modules.hardware.led import LedDriver
from modules.hardware.motor import MotorDriver
from sorter_app.state_machine import SorterConfig, SorterState

logger = logging.getLogger(__name__)

# Pin definition (BCM)
PIN_SWITCH = 17  # Physical Pin 11


class LegoSorter:
    """Main controller for the Lego Sorter.

    Manages the state machine (IDLE -> PREP -> SCAN -> SORT -> DONE)
    and coordinates LED, motor, and button hardware.
    """

    def __init__(self) -> None:
        self.state = SorterState.IDLE

        logger.info("Initializing Logic...")
        self.led = LedDriver()
        self.motor = MotorDriver()
        self.button = Button(PIN_SWITCH, pull_up=True, bounce_time=0.1)

        logger.info("System Ready.")

    def start_cycle(self) -> None:
        """Execute one full sorting cycle.

        Transitions through PREP -> SCAN -> SORT -> DONE -> IDLE.
        Only starts if current state is IDLE.
        """
        if self.state != SorterState.IDLE:
            return

        logger.info("Cycle Started")

        # 1. PREP: Illuminate
        self.state = SorterState.RUNNING
        self.led.fade_in(duration=SorterConfig.fade_duration)

        # 2. SCAN: Capture/Processing (simulated delay)
        logger.info("Scanning...")
        time.sleep(2.0)

        # 3. SORT: Action
        logger.info("Sorting...")
        self.motor.run_for(duration=SorterConfig.action_duration)

        # 4. DONE: Cleanup
        self.led.fade_out(duration=SorterConfig.fade_duration)

        # Return to IDLE
        self.state = SorterState.IDLE
        logger.info("Cycle Complete. Ready.")

    def update(self) -> None:
        """Main loop update called repeatedly."""
        if self.state == SorterState.IDLE:
            if self.button.is_pressed:
                logger.info("Button Triggered!")
                try:
                    self.start_cycle()
                except Exception as e:
                    logger.error("Error in cycle: %s", e)
                    self.cleanup()

    def cleanup(self) -> None:
        """Release all hardware resources safely."""
        logger.info("Cleaning up...")
        if self.led:
            self.led.cleanup()
        if self.motor:
            self.motor.cleanup()

    def run(self) -> None:
        """Entry point for the event loop."""
        logger.info("Entering Main Loop (Press Ctrl+C to exit)...")
        try:
            while True:
                self.update()
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
        finally:
            self.cleanup()
