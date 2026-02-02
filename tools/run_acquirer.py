import os
import sys
import time
import argparse
import logging
from datetime import datetime

# Add project root to path to import lego_sorter and sorter_app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from sorter_app.data.database_manager import DatabaseManager
    from lego_sorter import MotorDriver, LedDriver
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")
    raise

# Try importing OpenCV for camera
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


class ImageAcquirer:
    def __init__(self, db_path, set_num, motor=None, led=None, camera_index=0):
        self.db = DatabaseManager(db_path)
        self.set_num = set_num
        self.motor = motor if motor else MotorDriver()
        self.led = led if led else LedDriver()
        self.camera_index = camera_index
        self.cap = None
        self.logger = logging.getLogger("ImageAcquirer")

    def setup_camera(self):
        if not HAS_CV2:
            self.logger.warning("OpenCV not found. Camera capture will be simulated.")
            return False

        self.logger.info(f"Opening camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.logger.error("Failed to open camera.")
            return False

        # Warmup
        time.sleep(2)
        return True

    def capture_part(self, part_num, folder_name):
        """Capture images for a single part from multiple angles."""
        save_dir = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "raw", folder_name
        )
        os.makedirs(save_dir, exist_ok=True)

        self.logger.info(f"Starting capture for part {part_num}. Saving to {save_dir}")
        print(f"\n--- Capturing Part: {part_num} ---")
        print(f"Press Enter to start sequence for {part_num} (or 's' to skip)...")

        user_input = input().strip().lower()
        if user_input == "s":
            return False

        self.led.on()
        time.sleep(0.5)

        # Config: 8 angles (45 degrees each)
        num_angles = 8
        steps_per_angle = (
            512 // num_angles
        )  # 4096 steps per rev / 8 = 512 approx (28BYJ-48 is 4096 steps/rev half-step)
        # Verify lego_sorter.py step sequence/gearing.
        # Standard 28BYJ-48 is often ~4096 half-steps per rev.
        # We will assume 1 full rotation = 4096 steps.

        for i in range(num_angles):
            filename = f"{part_num}_{i}.jpg"
            filepath = os.path.join(save_dir, filename)

            # Capture
            if HAS_CV2 and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    cv2.imwrite(filepath, frame)
                    print(f"Captured: {filename}")
                else:
                    self.logger.error("Failed to read frame")
            else:
                # Simulate
                with open(filepath, "w") as f:
                    f.write("simulation_image")
                print(f"[SIM] Captured: {filename}")

            # Rotate
            # Using run_for as abstraction from lego_sorter or step if available.
            # lego_sorter.MotorDriver has step() and run_for().
            # Let's use step() for precision if available, but lego_sorter.py shows step() implementation.
            # 4096 steps / 8 = 512 steps
            self.motor.step(512)
            time.sleep(0.5)  # Stabilize

        self.led.off()
        return True

    def run(self):
        if HAS_CV2:
            if not self.setup_camera():
                print("Camera setup failed. Proceeding in simulation mode? (y/n)")
                if input().lower() != "y":
                    return

        unphotographed = self.db.get_unphotographed_parts(self.set_num)
        total = len(unphotographed)
        print(f"Found {total} unphotographed parts in set {self.set_num}")

        for idx, (part_num, name, color_id, color_name, _) in enumerate(unphotographed):
            print(
                f"\n[{idx+1}/{total}] Part: {part_num} | Name: {name} | Color: {color_name}"
            )

            # Create a unique folder name: part_num_color_id
            folder_name = f"{part_num}_{color_id}"

            success = self.capture_part(part_num, folder_name)

            if success:
                # Update DB
                # Path relative to dataset root usually preferred
                db_folder_path = f"raw/{folder_name}"
                self.db.update_part_image_folder(part_num, db_folder_path)
                print(f"Database updated for {part_num}")
            else:
                print(f"Skipped {part_num}")

        if self.cap:
            self.cap.release()
        self.motor.cleanup()
        self.led.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Lego Sorter Image Acquisition Tool")
    parser.add_argument(
        "--set",
        type=str,
        default="45345-1",
        help="Set Number to process (default: Spike Essential 45345-1)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sorter_app/data/lego_parts.sqlite",
        help="Path to database",
    )

    args = parser.parse_args()

    # Path correction for executing from root
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.db))

    acquirer = ImageAcquirer(db_path, args.set)
    acquirer.run()
