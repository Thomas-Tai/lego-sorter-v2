import time
import signal
import sys
from dataclasses import dataclass

try:
    from gpiozero import Button, PWMLED, OutputDevice

    # from gpiozero.pins.pigpio import PiGPIOFactory  # Pi 5 uses lgpio by default
except ImportError as e:
    print(f"WARNING: Hardware libraries not found ({e}). Using dummy classes.")

    # Fallback for non-Pi environments (e.g. CI/Windows dev)
    # We create dummy classes to avoid runtime errors during import
    # The actual functional logic should be mocked in tests
    class Button:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def is_pressed(self):
            return False

    class PWMLED:
        def __init__(self, *args, **kwargs):
            self.value = 0

        def on(self):
            pass

        def off(self):
            pass

    class OutputDevice:
        def __init__(self, *args, **kwargs):
            pass

        def on(self):
            pass

        def off(self):
            pass


# Constants
PIN_SWITCH = 17  # Physical Pin 11
PIN_LED = 11  # Physical Pin 23
PIN_MOTOR_IN1 = 27  # Physical Pin 13
PIN_MOTOR_IN2 = 22  # Physical Pin 15
PIN_MOTOR_IN3 = 10  # Physical Pin 19
PIN_MOTOR_IN4 = 9  # Physical Pin 21


@dataclass
class SorterConfig:
    fade_duration: float = 0.5
    action_duration: float = 5.0
    motor_delay: float = 0.002


# Motor Sequence (Half-step 8-phase)
STEP_SEQUENCE = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]


class MotorDriver:
    """Controls the 28BYJ-48 Stepper Motor via ULN2003."""

    def __init__(self):
        # Initialize pins [27, 22, 10, 9] (IN1..IN4)
        # Note: 10 & 9 are SPI pins being reused as GPIO
        self.pins = [
            OutputDevice(PIN_MOTOR_IN1),
            OutputDevice(PIN_MOTOR_IN2),
            OutputDevice(PIN_MOTOR_IN3),
            OutputDevice(PIN_MOTOR_IN4),
        ]

    def step(self, steps: int, direction: int = 1):
        """
        Rotate the motor by a number of steps.
        direction: 1 for CW, -1 for CCW
        """
        seq_len = len(STEP_SEQUENCE)
        for i in range(steps):
            # Calculate step index based on direction
            step_idx = (i * direction) % seq_len
            current_step = STEP_SEQUENCE[step_idx]

            # Apply to pins
            for pin, val in zip(self.pins, current_step):
                if val:
                    pin.on()
                else:
                    pin.off()

            # Speed control
            time.sleep(SorterConfig.motor_delay)

    def run_for(self, duration: float, direction: int = 1):
        """Run motor for a specific duration in seconds."""
        # Approx steps calculation not needed loop time based,
        # but better to just loop on time
        start_time = time.time()
        step_idx = 0
        seq_len = len(STEP_SEQUENCE)

        while (time.time() - start_time) < duration:
            # Determine next step
            step_idx = (step_idx + direction) % seq_len
            current_step = STEP_SEQUENCE[step_idx]

            for pin, val in zip(self.pins, current_step):
                if val:
                    pin.on()
                else:
                    pin.off()

            time.sleep(SorterConfig.motor_delay)

        self.stop()

    def stop(self):
        """De-energize all coils to prevent overheating."""
        for pin in self.pins:
            pin.off()

    def cleanup(self):
        self.stop()


class LedDriver:
    """Controls the 12V LED Strip via MOSFET using PWM."""

    def __init__(self, pin: int = PIN_LED):
        # 100Hz frequency for stability as verified in hardware smoke test
        self.led = PWMLED(pin, frequency=100)

    def on(self):
        """Turn LED fully on."""
        self.led.on()

    def off(self):
        """Turn LED fully off."""
        self.led.off()

    def fade_in(self, duration: float = 0.5):
        """Fade LED from 0% to 100% using Gamma correction."""
        steps = 20
        delay = duration / steps
        for i in range(steps + 1):
            lin = i / float(steps)
            # Gamma 2.8 encoding
            duty = pow(lin, 2.8)
            self.led.value = duty
            time.sleep(delay)

    def fade_out(self, duration: float = 0.5):
        """Fade LED from 100% to 0% using Gamma correction."""
        steps = 20
        delay = duration / steps
        for i in range(steps + 1):
            lin = (steps - i) / float(steps)
            duty = pow(lin, 2.8)
            self.led.value = duty
            time.sleep(delay)

    def cleanup(self):
        self.led.off()


from enum import Enum, auto


class SorterState(Enum):
    IDLE = auto()
    PREP = auto()
    SCAN = auto()
    SORT = auto()
    DONE = auto()


class LegoSorter:
    """
    Main controller for the Lego Sorter V4.5.
    Manages the state machine: IDLE -> PREP -> SCAN -> SORT -> DONE.
    """

    def __init__(self):
        self.state = SorterState.IDLE

        # Initialize Drivers
        print("Initializing Logic...")
        self.led = LedDriver()
        self.motor = MotorDriver()

        # Initialize Button
        # Pull-up=True means logic low when pressed
        self.button = Button(PIN_SWITCH, pull_up=True, bounce_time=0.1)

        print("System Ready.")

    def start_cycle(self):
        """Execute one full sorting cycle."""
        if self.state != SorterState.IDLE:
            return

        print("Cycle Started")

        # 1. PREP: Illuminate
        self.state = SorterState.PREP
        self.led.fade_in(duration=SorterConfig.fade_duration)

        # 2. SCAN: Capture/Processing (Simulated delay for now)
        self.state = SorterState.SCAN
        print("Scanning...")
        time.sleep(2.0)  # Placeholder for camera logic

        # 3. SORT: Action
        self.state = SorterState.SORT
        print("Sorting...")
        self.motor.run_for(duration=SorterConfig.action_duration)

        # 4. DONE: Cleanup
        self.state = SorterState.DONE
        self.led.fade_out(duration=SorterConfig.fade_duration)

        # Return to IDLE
        self.state = SorterState.IDLE
        print("Cycle Complete. Ready.")

    def update(self):
        """Main loop update called repeatedly."""
        if self.state == SorterState.IDLE:
            if self.button.is_pressed:
                print("Button Triggered!")
                try:
                    self.start_cycle()
                except Exception as e:
                    print(f"Error in cycle: {e}")
                    self.cleanup()
        else:
            # If we are in a blocking state, start_cycle handles it.
            # Ideally start_cycle should be non-blocking or threaded,
            # but for V1 simple script, blocking is acceptable as per requirements.
            pass

    def cleanup(self):
        """Safe shutdown."""
        print("Cleaning up...")
        if self.led:
            self.led.cleanup()
        if self.motor:
            self.motor.cleanup()

    def run(self):
        """Entry point for the event loop."""
        print("Entering Main Loop (Press Ctrl+C to exit)...")
        try:
            while True:
                self.update()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nShutdown requested.")
        finally:
            self.cleanup()


if __name__ == "__main__":
    sorter = LegoSorter()
    sorter.run()
