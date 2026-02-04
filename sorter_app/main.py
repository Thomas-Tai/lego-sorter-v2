import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sorter_app.services.config_service import ConfigService
from sorter_app.services.api_client import APIClient
from modules.hardware.camera import CameraDriver
from modules.hardware.led import LedDriver

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SorterApp")


def main():
    logger.info("Starting Lego Sorter App...")

    # Parse CLI args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-image", type=str, help="Path to test image (skip capture)"
    )
    args = parser.parse_args()

    # 1. Initialize Services
    config_service = ConfigService()
    api_client = APIClient(base_url=config_service.api_url)

    # 2. Main Logic
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, "data")
        image_path = os.path.join(data_dir, "captures", "test_capture.jpg")

        captured = False

        if args.test_image:
            logger.info(f"Using test image: {args.test_image}")
            image_path = args.test_image
            if not os.path.exists(image_path):
                logger.error(f"Test image not found: {image_path}")
                return
            captured = True
        else:
            # Initialize hardware
            camera = CameraDriver(camera_index=config_service.camera_index)
            led = LedDriver()
            try:
                if not camera.open():
                    logger.error("Failed to open camera. Check connection.")
                    return

                # Turn on LED for consistent lighting (same as acquisition)
                led.on()
                logger.info("LED on for consistent lighting")
                time.sleep(0.3)  # Allow LED to stabilize

                logger.info("Capturing image...")
                # Ensure directory exists
                Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                if camera.capture(image_path):
                    captured = True
                    logger.info(f"Image captured to {image_path}")
                else:
                    logger.error("Failed to capture image.")
            finally:
                camera.close()
                led.cleanup()
                logger.info("LED off, hardware cleanup complete")

        if captured:
            logger.info("Sending to inference API...")
            try:
                result = api_client.predict_from_image(image_path)

                if result.get("success"):
                    matches = result.get("matches", [])
                    if matches:
                        top_match = matches[0]
                        logger.info(
                            f"✅ IDENTIFIED: {top_match['part_id']} (Color: {top_match['color_id']})"
                        )
                        logger.info(f"   Confidence: {top_match['confidence']}")
                        logger.info(f"   Source: {top_match['source']}")
                    else:
                        logger.info("❓ No matches found.")
                else:
                    logger.error(f"API Error: {result}")

            except IOError as e:
                logger.error(f"Prediction failed: {e}")

        else:
            logger.error("Failed to capture image.")

    except KeyboardInterrupt:
        logger.info("Stopping app...")


if __name__ == "__main__":
    main()
