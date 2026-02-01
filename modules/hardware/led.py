"""
LED Driver Module
Provides LedDriver class for PWM LED strip control.
"""
import time

try:
    from gpiozero import PWMLED
except ImportError:
    # Fallback for non-Pi environments
    class PWMLED:
        def __init__(self, *args, **kwargs):
            self.value = 0

        def on(self):
            pass

        def off(self):
            pass


# Pin Definition (BCM)
PIN_LED = 11  # Physical Pin 23


class LedDriver:
    """Controls the 12V LED Strip via MOSFET using PWM."""

    def __init__(self, pin: int = PIN_LED, frequency: int = 100):
        self.led = PWMLED(pin, frequency=frequency)

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
