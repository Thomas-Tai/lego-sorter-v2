"""
Motor Driver Module
Provides MotorDriver class for 28BYJ-48 stepper motor control.
"""

import time

try:
    from gpiozero import OutputDevice
except ImportError:
    # Fallback for non-Pi environments
    class OutputDevice:
        def __init__(self, *args, **kwargs):
            pass

        def on(self):
            pass

        def off(self):
            pass


# Pin Definitions (BCM)
PIN_MOTOR_IN1 = 27  # Physical Pin 13
PIN_MOTOR_IN2 = 22  # Physical Pin 15
PIN_MOTOR_IN3 = 10  # Physical Pin 19
PIN_MOTOR_IN4 = 9  # Physical Pin 21

# Motor Sequence (Half-step 8-phase for smoothness)
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

# Default motor delay (speed control) - 1.5ms for reliable stepping
DEFAULT_MOTOR_DELAY = 0.0015


class MotorDriver:
    """Controls the 28BYJ-48 Stepper Motor via ULN2003."""

    def __init__(self, motor_delay: float = DEFAULT_MOTOR_DELAY):
        self.motor_delay = motor_delay
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
            step_idx = (i * direction) % seq_len
            current_step = STEP_SEQUENCE[step_idx]

            for pin, val in zip(self.pins, current_step):
                if val:
                    pin.on()
                else:
                    pin.off()

            time.sleep(self.motor_delay)

    def run_for(self, duration: float, direction: int = 1):
        """Run motor for a specific duration in seconds."""
        start_time = time.time()
        step_idx = 0
        seq_len = len(STEP_SEQUENCE)

        while (time.time() - start_time) < duration:
            step_idx = (step_idx + direction) % seq_len
            current_step = STEP_SEQUENCE[step_idx]

            for pin, val in zip(self.pins, current_step):
                if val:
                    pin.on()
                else:
                    pin.off()

            time.sleep(self.motor_delay)

        self.stop()

    def stop(self):
        """De-energize all coils to prevent overheating."""
        for pin in self.pins:
            pin.off()

    def cleanup(self):
        self.stop()
