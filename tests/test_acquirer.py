import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import module under test
try:
    import tools.run_acquirer
    from tools.run_acquirer import ImageAcquirer
except ImportError as e:
    print(f"Test Import Error: {e}")
    sys.exit(1)


class TestImageAcquirer(unittest.TestCase):
    def setUp(self):
        self.mock_motor = MagicMock()
        self.mock_led = MagicMock()

    @patch("tools.run_acquirer.cv2")
    @patch("tools.run_acquirer.DatabaseManager")
    @patch("builtins.input", return_value="")
    @patch("tools.run_acquirer.os.makedirs")
    def test_capture_cycle(self, mock_makedirs, mock_input, MockDB, mock_cv2):
        print("Starting test_capture_cycle")

        # Setup Mock DB
        mock_db_instance = MockDB.return_value
        mock_db_instance.get_unphotographed_parts.return_value = [
            ("3001", "Brick 2x4", 15, "White", None)
        ]

        # Init Acquirer
        # Trigger import/init
        acquirer = ImageAcquirer(
            "dummy.sqlite", "1234-1", self.mock_motor, self.mock_led
        )

        # Mock Camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "fake_frame")

        # Inject mock camera into acquirer (since setup_camera might try to open real one or fail)
        # We can bypass setup_camera by setting acquirer.cap directly and HAS_CV2 logic
        # But run() calls setup_camera().

        # Let's mock cv2.VideoCapture to return our mock_cap
        mock_cv2.VideoCapture.return_value = mock_cap

        # Ensure HAS_CV2 is True in the module (it might be False if import failed in module)
        tools.run_acquirer.HAS_CV2 = True

        # Run
        try:
            acquirer.run()
        except Exception as e:
            self.fail(f"acquirer.run() raised {e}")

        # Verification
        self.mock_led.on.assert_called()
        self.mock_led.off.assert_called()

        # 8 steps
        self.assertEqual(self.mock_motor.step.call_count, 8)

        # 8 writes
        self.assertEqual(mock_cv2.imwrite.call_count, 8)

        # DB Update
        mock_db_instance.update_part_image_folder.assert_called_with(
            "3001", "raw/3001_15"
        )
        print("Test passed verification")


if __name__ == "__main__":
    unittest.main()
