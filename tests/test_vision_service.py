"""Unit tests for the VisionService."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sorter_app.exceptions import CameraError
from sorter_app.services.vision_service import RaspberryPiVisionService


class TestRaspberryPiVisionService(unittest.TestCase):
    """Tests for RaspberryPiVisionService."""

    @patch("sorter_app.services.vision_service.cv2.VideoCapture")
    def test_init_opens_camera(self, mock_capture_class: MagicMock) -> None:
        """Test that __init__ opens the camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap

        service = RaspberryPiVisionService(camera_index=0)

        mock_capture_class.assert_called_once_with(0)
        mock_cap.isOpened.assert_called_once()
        self.assertIsNotNone(service)

    @patch("sorter_app.services.vision_service.cv2.VideoCapture")
    def test_init_raises_on_camera_failure(self, mock_capture_class: MagicMock) -> None:
        """Test that __init__ raises CameraError if camera fails to open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap

        with self.assertRaises(CameraError):
            RaspberryPiVisionService(camera_index=0)

    @patch("sorter_app.services.vision_service.cv2.imwrite")
    @patch("sorter_app.services.vision_service.cv2.VideoCapture")
    def test_capture_image_success(self, mock_capture_class: MagicMock, mock_imwrite: MagicMock) -> None:
        """Test successful image capture."""
        # Setup mocks
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        mock_capture_class.return_value = mock_cap
        mock_imwrite.return_value = True

        # Create service and capture
        service = RaspberryPiVisionService()
        result = service.capture_image("/tmp/test.jpg")

        # Verify
        self.assertTrue(result)
        mock_cap.read.assert_called_once()
        mock_imwrite.assert_called_once()

    @patch("sorter_app.services.vision_service.cv2.VideoCapture")
    def test_capture_image_read_failure(self, mock_capture_class: MagicMock) -> None:
        """Test that capture_image returns False when read fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_capture_class.return_value = mock_cap

        service = RaspberryPiVisionService()
        result = service.capture_image("/tmp/test.jpg")

        self.assertFalse(result)

    @patch("sorter_app.services.vision_service.cv2.VideoCapture")
    def test_release_closes_camera(self, mock_capture_class: MagicMock) -> None:
        """Test that release() closes the camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap

        service = RaspberryPiVisionService()
        service.release()

        mock_cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
