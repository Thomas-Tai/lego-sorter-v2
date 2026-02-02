"""
Image Acquirer Module
Provides ImageAcquirer class for capturing part images from multiple angles.

Folder Structure:
    data/images/raw/{part_num}/{color_id}/{part_num}_{color_id}_{angle}.jpg

Manifest:
    data/images/manifest.csv - CSV with image_path, part_num, color_id, color_name, timestamp
"""

import os
import csv
import time
import logging
from datetime import datetime

from modules.database import DatabaseManager
from modules.hardware import MotorDriver, LedDriver, CameraDriver, ButtonDriver

# Default paths
DEFAULT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "images")
MANIFEST_FILENAME = "manifest.csv"


class ImageAcquirer:
    """Captures training images for LEGO parts using rotary platform."""

    def __init__(
        self,
        db_path: str = None,
        set_num: str = "45345-1",
        motor: MotorDriver = None,
        led: LedDriver = None,
        camera: CameraDriver = None,
        images_dir: str = None,
    ):
        self.db = DatabaseManager(db_path)
        self.set_num = set_num
        self.motor = motor if motor else MotorDriver()
        self.led = led if led else LedDriver()
        self.camera = camera if camera else CameraDriver()
        self.button = ButtonDriver()  # Physical button for triggering capture
        self.images_dir = images_dir if images_dir else DEFAULT_IMAGES_DIR
        self.raw_dir = os.path.join(self.images_dir, "raw")
        self.manifest_path = os.path.join(self.images_dir, MANIFEST_FILENAME)
        self.logger = logging.getLogger("ImageAcquirer")

        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)

        # Initialize manifest if it doesn't exist
        self._init_manifest()

    def _init_manifest(self):
        """Initialize manifest.csv with headers if it doesn't exist."""
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "part_num", "color_id", "color_name", "part_name", "angle", "timestamp"])
            self.logger.info(f"Created manifest at {self.manifest_path}")

    def _append_to_manifest(
        self, image_path: str, part_num: str, color_id: int, color_name: str, part_name: str, angle: int
    ):
        """Append a captured image entry to the manifest."""
        # Store relative path from images_dir
        relative_path = os.path.relpath(image_path, self.images_dir)

        with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [relative_path, part_num, color_id, color_name, part_name, angle, datetime.now().isoformat()]
            )

    def _capture_one_face(
        self,
        save_dir: str,
        part_num: str,
        color_id: int,
        color_name: str,
        part_name: str,
        start_angle: int,
        face_label: str,
    ) -> int:
        """Capture 8 angles for one face of the part.

        Returns the number of images captured.
        """
        import sys
        import select

        num_angles = 8
        steps_per_angle = 2048
        captured = 0

        print(f"\n--- 拍攝{face_label} (角度 {start_angle}-{start_angle+7}) ---")
        for i in range(num_angles):
            angle_idx = start_angle + i
            filename = f"{part_num}_{color_id}_{angle_idx}.jpg"
            filepath = os.path.join(save_dir, filename)

            if self.camera.capture(filepath):
                print(f"  [{face_label}] Captured: {filename}")
            else:
                print(f"  [SIM] Captured: {filename}")

            self._append_to_manifest(filepath, part_num, color_id, color_name, part_name, angle_idx)
            captured += 1
            self.motor.step(steps_per_angle)
            time.sleep(0.5)

        return captured

    def _wait_for_input(self, skip_allowed: bool = True) -> str:
        """Wait for button press or keyboard input.

        Returns: 'continue', 'skip', or the input string
        """
        import sys
        import select

        while True:
            if self.button.is_pressed():
                self.button.wait_for_press(timeout=0.1)
                return "continue"
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == "s" and skip_allowed:
                        return "skip"
                    return user_input if user_input else "continue"
            except Exception:
                pass
            time.sleep(0.05)

    def capture_part(self, part_num: str, color_id: int, color_name: str, part_name: str) -> bool:
        """Capture images for a single part with selectable capture modes.

        Modes:
          1. Simple (16 images): Top + Bottom
          2. Standard (24 images): Top + Bottom + Side1
          3. Complete (32 images): Top + Bottom + Side1 + Side2
        """
        save_dir = os.path.join(self.raw_dir, str(part_num), str(color_id))
        os.makedirs(save_dir, exist_ok=True)

        self.logger.info(f"Starting capture for {part_num} (color: {color_name})")
        print(f"\n{'='*60}")
        print(f"  Part: {part_num}")
        print(f"  Color: {color_name} (ID: {color_id})")
        print(f"  Reference: https://rebrickable.com/parts/{part_num}/")
        print(f"  Image: https://cdn.rebrickable.com/media/parts/photos/{color_id}/{part_num}.jpg")
        print(f"{'='*60}")
        print(f"\n選擇拍攝模式:")
        print(f"  1 = 簡單 (16張: 頂+底) - 對稱零件")
        print(f"  2 = 標準 (24張: 頂+底+側) - 一般零件")
        print(f"  3 = 完整 (32張: 頂+底+2側) - 複雜零件")
        print(f"  s = 跳過此零件")
        print(f"\n按鈕 = 模式1 | 輸入 1/2/3/s + Enter")

        # Get capture mode
        mode_input = self._wait_for_input(skip_allowed=True)
        if mode_input == "skip":
            return False

        # Determine number of faces based on mode
        if mode_input == "3":
            num_faces = 4
            mode_name = "完整"
        elif mode_input == "2":
            num_faces = 3
            mode_name = "標準"
        else:
            num_faces = 2
            mode_name = "簡單"

        print(f"\n>>> 模式: {mode_name} ({num_faces * 8}張)")

        face_labels = ["正面", "反面", "側面1", "側面2"]
        captured_count = 0

        for face_idx in range(num_faces):
            if face_idx > 0:
                print(f"\n{'='*60}")
                print(f"  [第{face_idx+1}輪] 請將零件{face_labels[face_idx]}朝上")
                print(f"  按鈕繼續... (或 's' 結束拍攝)")
                print(f"{'='*60}")

                inp = self._wait_for_input(skip_allowed=True)
                if inp == "skip":
                    self.logger.info(f"Captured {captured_count} images for {part_num}")
                    return True
            else:
                print(f"\n[第1輪] 零件{face_labels[0]}朝上 - 按鈕開始...")
                self._wait_for_input(skip_allowed=False)

            time.sleep(0.3)
            start_angle = face_idx * 8
            captured_count += self._capture_one_face(
                save_dir, part_num, color_id, color_name, part_name, start_angle, face_labels[face_idx]
            )

        self.logger.info(f"Captured {captured_count} images for {part_num}")
        return True

    def run(self):
        """Run the acquisition workflow for unphotographed parts."""
        if not self.camera.open():
            print("Camera not available. Running in simulation mode.")

        # LED on for entire session (consistent lighting)
        self.led.on()
        print("LED on for consistent lighting")

        unphotographed = self.db.get_unphotographed_parts(self.set_num)
        total = len(unphotographed)
        print(f"Found {total} unphotographed parts in set {self.set_num}")
        print(f"Manifest: {self.manifest_path}")

        for idx, (part_num, name, color_id, color_name, _) in enumerate(unphotographed):
            print(f"\n[{idx+1}/{total}] Part: {part_num} | Name: {name} | Color: {color_name}")

            success = self.capture_part(part_num, color_id, color_name, name)

            if success:
                # Update DB with hierarchical path: raw/{part_num}/{color_id}
                db_folder_path = f"raw/{part_num}/{color_id}"
                self.db.update_part_image_folder(part_num, db_folder_path)
                print(f"Database updated for {part_num}")
            else:
                print(f"Skipped {part_num}")

        self.camera.close()
        self.motor.cleanup()
        self.led.cleanup()
        self.button.cleanup()

        print(f"\nAcquisition complete. Manifest saved to: {self.manifest_path}")
