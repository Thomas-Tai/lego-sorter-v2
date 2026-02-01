"""
Button Driver Module
Provides ButtonDriver class for physical button input on Raspberry Pi.
"""
try:
    from gpiozero import Button
    HAS_GPIO = True
except ImportError:
    Button = None
    HAS_GPIO = False


class ButtonDriver:
    """Handles physical button input."""

    def __init__(self, pin: int = 17):
        """Initialize button on specified GPIO pin.
        
        Args:
            pin: BCM GPIO pin number (default 17 = physical pin 11)
        """
        self.pin = pin
        self.button = None
        
        if HAS_GPIO:
            try:
                self.button = Button(pin, pull_up=True, bounce_time=0.1)
            except Exception:
                self.button = None

    def wait_for_press(self, timeout: float = None) -> bool:
        """Wait for button press. Returns True if pressed, False if timeout.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
        """
        if not HAS_GPIO or self.button is None:
            # Simulation mode - return immediately
            return True

        if timeout is None:
            self.button.wait_for_press()
            return True
        else:
            return self.button.wait_for_press(timeout=timeout)

    def is_pressed(self) -> bool:
        """Check if button is currently pressed."""
        if not HAS_GPIO or self.button is None:
            return False
        return self.button.is_pressed

    def cleanup(self):
        """Release button resources."""
        if self.button is not None:
            self.button.close()
            self.button = None
