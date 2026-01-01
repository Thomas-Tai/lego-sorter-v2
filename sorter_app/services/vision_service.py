"""Vision service implementation for Raspberry Pi USB camera."""

import logging
from pathlib import Path

import cv2

from .base_service import AbstractVisionService
from ..exceptions import CameraError

logger = logging.getLogger(__name__)


class RaspberryPiVisionService(AbstractVisionService):
    """Vision service implementation using OpenCV for USB camera capture.

    Attributes:
        camera_index: The device index of the camera (default 0).
    """

    def __init__(self, camera_index: int = 0) -> None:
        """Initializes the vision service and opens the camera.

        Args:
            camera_index: The device index of the USB camera.

        Raises:
            CameraError: If the camera cannot be opened.
        """
        self._camera_index = camera_index
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise CameraError(
                f"Failed to open camera at index {camera_index}. "
                "Ensure the camera is connected and permissions are set."
            )
        logger.info("Camera opened successfully at index %d", camera_index)

    def capture_image(self, filepath: str) -> bool:
        """Captures an image and saves it to the given path.

        Args:
            filepath: Absolute path where the image should be saved.

        Returns:
            True if successful, False otherwise.
        """
        # Flush the camera buffer by reading and discarding frames.
        # USB cameras buffer frames internally, causing stale images if there's
        # a delay between captures. We discard several frames to ensure we get
        # the current view.
        buffer_flush_count = 5
        for _ in range(buffer_flush_count):
            self._cap.grab()  # Grab without decoding for speed

        # Now capture the actual frame
        ret, frame = self._cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return False

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(filepath, frame)
        if success:
            logger.debug("Image saved to %s", filepath)
        else:
            logger.error("Failed to write image to %s", filepath)
        return success

    def release(self) -> None:
        """Releases the camera resource."""
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera released")
