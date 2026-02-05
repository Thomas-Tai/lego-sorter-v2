"""
Camera Driver Module
Provides CameraDriver class for webcam/USB camera control.
"""

import os
import time

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


class CameraDriver:
    """Handles camera capture operations."""

    def __init__(self, camera_index: int = 0, buffer_flush_count: int = 5):
        self.camera_index = camera_index
        self.buffer_flush_count = buffer_flush_count  # Frames to discard before capture
        self.cap = None

    def open(self) -> bool:
        """Open the camera. Returns True if successful."""
        if not HAS_CV2:
            return False

        # Use CAP_DSHOW on Windows to avoid MSMF errors/black screens
        if os.name == "nt":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            # FORCE MJPG (Fixes static/corrupted frames on many USB cameras)
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )

            # Set camera properties for better image quality
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer

            # Set resolution to match training images (1280x720)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Enable autofocus (if supported by camera)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            # Disable auto-exposure for consistent lighting (LED provides stable light)
            # Note: Some cameras may not support this
            # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode

            # Log actual resolution obtained
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera opened: {actual_w}x{actual_h}")

            return True
        return False

    def _flush_buffer(self):
        """Discard buffered frames to get the current live image."""
        if self.cap is None:
            return
        # Increase flush count significantly to warm up auto-exposure (Guided by docs)
        for _ in range(30):
            self.cap.grab()  # Discard frame without decoding

    def capture(self, filepath: str) -> bool:
        """Capture a single frame and save to filepath. Returns True if successful.

        Flushes buffer first to ensure we get the current live image.
        """
        if not HAS_CV2 or self.cap is None or not self.cap.isOpened():
            # Simulation mode
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write("simulation_image")
            return False  # Return False to indicate simulation

        # Flush buffer to get current frame
        self._flush_buffer()

        # Small delay for camera to stabilize
        time.sleep(0.05)

        ret, frame = self.cap.read()
        if ret and frame is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv2.imwrite(filepath, frame)
            return True
        return False

    def close(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_open(self) -> bool:
        """Check if camera is open."""
        if self.cap is None:
            return False
        return self.cap.isOpened()
