"""Sorter application entry point with dependency injection."""

import argparse
import logging
import os
import time
from pathlib import Path

from sorter_app.services.config_service import ConfigService
from sorter_app.services.api_client import APIClient
from sorter_app.services.hardware_service import RaspberryPiHardwareService
from sorter_app.services.vision_service import RaspberryPiVisionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SorterApp")


def main() -> None:
    """Run the sorter application main loop.

    Parses CLI arguments, initializes services via DI, captures an image
    (or uses a provided test image), and sends it to the inference API.
    """
    logger.info("Starting Lego Sorter App...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-image", type=str, help="Path to test image (skip capture)"
    )
    args = parser.parse_args()

    config_service = ConfigService()
    api_client = APIClient(base_url=config_service.api_url)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, "data")
        image_path = os.path.join(data_dir, "captures", "test_capture.jpg")

        captured = False

        if args.test_image:
            logger.info("Using test image: %s", args.test_image)
            image_path = args.test_image
            if not os.path.exists(image_path):
                logger.error("Test image not found: %s", image_path)
                return
            captured = True
        else:
            hardware_service = RaspberryPiHardwareService()
            vision_service = RaspberryPiVisionService(
                camera_index=config_service.camera_index,
            )
            try:
                hardware_service.set_led_power(True)
                logger.info("LED on for consistent lighting")
                time.sleep(0.3)

                logger.info("Capturing image...")
                Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                if vision_service.capture_image(image_path):
                    captured = True
                    logger.info("Image captured to %s", image_path)
                else:
                    logger.error("Failed to capture image.")
            finally:
                vision_service.release()
                hardware_service.cleanup()
                logger.info("Hardware cleanup complete")

        if captured:
            logger.info("Sending to inference API...")
            try:
                result = api_client.predict_from_image(image_path)

                if result.get("success"):
                    matches = result.get("matches", [])
                    if matches:
                        top_match = matches[0]
                        logger.info(
                            "IDENTIFIED: %s (Color: %s)",
                            top_match["part_id"],
                            top_match["color_id"],
                        )
                        logger.info("   Confidence: %s", top_match["confidence"])
                        logger.info("   Source: %s", top_match["source"])
                    else:
                        logger.info("No matches found.")
                else:
                    logger.error("API Error: %s", result)

            except IOError as e:
                logger.error("Prediction failed: %s", e)

        else:
            logger.error("Failed to capture image.")

    except KeyboardInterrupt:
        logger.info("Stopping app...")


if __name__ == "__main__":
    main()
