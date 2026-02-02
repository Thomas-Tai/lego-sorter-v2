import time
import sys
import signal

# Try to import gpiozero, fallback to mock if on PC
try:
    from gpiozero import Button, PWMLED, OutputDevice
    from gpiozero.pins.rpigpio import RPiGPIOFactory
except ImportError:
    print("⚠️  gpiozero not found! Using mock objects for testing logic on PC.")

    class Button:
        def __init__(self, pin, pull_up=True, bounce_time=None):
            self.pin = pin

        def wait_for_press(self, timeout=None):
            time.sleep(0.5)
            return True

        def is_pressed(self):
            return False

    class PWMLED:
        def __init__(self, pin):
            self.pin = pin

        def value(self, v):
            print(f"[MOCK] LED {self.pin} val={v}")

        def on(self):
            print(f"[MOCK] LED {self.pin} ON")

        def off(self):
            print(f"[MOCK] LED {self.pin} OFF")

    class OutputDevice:
        def __init__(self, pin):
            self.pin = pin

        def on(self):
            print(f"[MOCK] Pin {self.pin} HI")

        def off(self):
            print(f"[MOCK] Pin {self.pin} LO")


# --- V4.5 PIN DEFINITIONS (BCM) ---
PIN_SWITCH = 17  # Physical Pin 11
PIN_LED = 11  # Physical Pin 23
PIN_MOTOR_IN1 = 27  # Physical Pin 13
PIN_MOTOR_IN2 = 22  # Physical Pin 15
PIN_MOTOR_IN3 = 10  # Physical Pin 19 (SPI MOSI Override)
PIN_MOTOR_IN4 = 9  # Physical Pin 21 (SPI MISO Override)


def test_led():
    print("\n--- Testing LED (Pin 23 / GPIO 11) ---")
    print("Action: Blinking and Fading. Check Signal/Voltage now.")
    # Try 100Hz for better compatibility with MOSFETs/Software PWM
    led = PWMLED(PIN_LED, initial_value=0, frequency=100)

    # Blink
    print(">> Blinking (Full On/Off)...")
    for _ in range(3):
        led.value = 1.0
        time.sleep(0.5)
        led.value = 0.0
        time.sleep(0.5)

    # Fade (Gamma Corrected for human eye)
    print(">> Fading (Gamma Corrected for smoothness)...")

    # Gamma 2.8 curve
    steps = 50
    for i in range(steps + 1):
        # Linear (0.0 to 1.0)
        lin = i / float(steps)
        # Gamma encoding (approx 2.8)
        duty = pow(lin, 2.8)

        led.value = duty
        # Visualize bar chart
        bar = "#" * int(lin * 20)
        print(f"   PWM: {int(duty*100):3d}% [{bar:<20}]", end="\r")
        time.sleep(0.05)

    print("\n   Hold Full Brightness...")
    time.sleep(1)

    for i in range(steps + 1):
        lin = (steps - i) / float(steps)
        duty = pow(lin, 2.8)
        led.value = duty
        bar = "#" * int(lin * 20)
        print(f"   PWM: {int(duty*100):3d}% [{bar:<20}]", end="\r")
        time.sleep(0.05)

    print("\n")
    led.off()
    print(">> LED Test Complete.")


def test_motor():
    print("\n--- Testing Motor (Pins 13,15,19,21) ---")
    print("Action: Rotating. Check Coil Voltages.")

    # Initialize Motor Pins
    m1 = OutputDevice(PIN_MOTOR_IN1)
    m2 = OutputDevice(PIN_MOTOR_IN2)
    m3 = OutputDevice(PIN_MOTOR_IN3)
    m4 = OutputDevice(PIN_MOTOR_IN4)
    motor_pins = [m1, m2, m3, m4]

    # 8-step sequence
    seq = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    print(">> Rotating CW (5 sec)...")
    start = time.time()
    while time.time() - start < 5:
        for step in seq:
            for pin, val in zip(motor_pins, step):
                if val:
                    pin.on()
                else:
                    pin.off()
            time.sleep(0.002)  # ~500Hz

    # Stop (De-energize)
    print(">> Stopping (All Low - Safety Check)...")
    for pin in motor_pins:
        pin.off()
    print(">> Motor Test Complete. Verify NO holding torque (shaft should spin freely).")


def test_button():
    print("\n--- Testing Button (Pin 11 / GPIO 17) ---")
    print("Action: Waiting for press. Please press the silver button.")
    btn = Button(PIN_SWITCH, pull_up=True, bounce_time=0.2)

    try:
        if btn.wait_for_press(timeout=10):
            print(">> SUCCESS: Button Press Detected!")
        else:
            print(">> TIMEOUT: Button not pressed (or signal missing).")
    except Exception as e:
        print(f">> ERROR: {e}")


def main():
    print("=== Lego Sorter V4.5 Hardware Smoke Test ===")
    print("WARNING: This script activates 12V components.")

    try:
        test_led()
        test_motor()
        test_button()
        print("\n=== Verification Complete ===")
    except KeyboardInterrupt:
        print("\nAborted.")


if __name__ == "__main__":
    main()
