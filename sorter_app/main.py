"""Sorter application entry point with dependency injection."""

import argparse
import logging
import os
import time
from pathlib import Path

import yaml

from sorter_app.services.config_service import ConfigService
from sorter_app.services.api_client import APIClient
from sorter_app.services.hardware_service import RaspberryPiHardwareService
from sorter_app.services.vision_service import RaspberryPiVisionService
from sorter_app.services.gantry_sorting_service import GantrySortingService
from sorter_app.domain.schemas import BinLayoutConfig, GantryConfig
from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.hardware import GantryClient, MockGantryClient
from sorter_app.exceptions import GantryError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SorterApp")

# Confidence threshold for sorting decision
CONFIDENCE_THRESHOLD = 0.80


def load_gantry_config(config_path: str) -> GantryConfig:
    """Load gantry configuration from YAML file.

    Args:
        config_path: Path to gantry.yaml.

    Returns:
        Validated GantryConfig.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return GantryConfig(**data["gantry"])


def load_bin_layout_config(config_path: str) -> BinLayoutConfig:
    """Load bin layout configuration from YAML file.

    Args:
        config_path: Path to bin_layout.yaml.

    Returns:
        Validated BinLayoutConfig.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return BinLayoutConfig(**data["bin_layout"])


def main() -> None:
    """Run the sorter application main loop.

    Parses CLI arguments, initializes services via DI, captures an image
    (or uses a provided test image), and sends it to the inference API.
    If --sort is specified, sorts the part to the appropriate bin.
    """
    logger.info("Starting Lego Sorter App...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-image", type=str, help="Path to test image (skip capture)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use mock gantry client (simulation mode, no hardware required)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Enable sorting (requires gantry hardware or --simulate)",
    )
    args = parser.parse_args()

    config_service = ConfigService()
    api_client = APIClient(base_url=config_service.api_url)

    # Determine config paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_dir = os.path.join(project_root, "config")

    # Initialize sorting services if sorting is enabled
    sorting_service: GantrySortingService | None = None
    bin_mapper: BinMapper | None = None

    if args.sort:
        try:
            # Load configurations
            gantry_config = load_gantry_config(os.path.join(config_dir, "gantry.yaml"))
            bin_layout_config = load_bin_layout_config(
                os.path.join(config_dir, "bin_layout.yaml")
            )

            # Create bin mapper
            bin_mapper = BinMapper(bin_layout_config)

            # Create gantry client (real or mock)
            if args.simulate:
                logger.info("Using MockGantryClient (simulation mode)")
                gantry_client = MockGantryClient(gantry_config)
            else:
                logger.info("Using GantryClient (hardware mode)")
                gantry_client = GantryClient(gantry_config)

            # Connect to gantry
            gantry_client.connect()
            gantry_client.home()

            # Create sorting service
            sorting_service = GantrySortingService(gantry_client, bin_mapper)

        except GantryError as e:
            logger.error("Failed to initialize sorting: %s", e)
            args.sort = False  # Disable sorting but continue with inference

    try:
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
                        part_id = top_match["part_id"]
                        color_id = top_match["color_id"]
                        confidence = top_match["confidence"]

                        logger.info(
                            "IDENTIFIED: %s (Color: %s)",
                            part_id,
                            color_id,
                        )
                        logger.info("   Confidence: %s", confidence)
                        logger.info("   Source: %s", top_match["source"])

                        # Sort to bin if sorting is enabled
                        if args.sort and sorting_service and bin_mapper:
                            if confidence < CONFIDENCE_THRESHOLD:
                                # Low confidence -> overflow bin
                                bin_id = bin_mapper.overflow_id
                                logger.info(
                                    "Low confidence (%.2f < %.2f), routing to overflow bin %d",
                                    confidence,
                                    CONFIDENCE_THRESHOLD,
                                    bin_id,
                                )
                            else:
                                # Get bin for part
                                bin_info = bin_mapper.get_bin_for_part(
                                    part_id, color_id
                                )
                                bin_id = bin_info.id
                                logger.info(
                                    "Routing to bin %d (%s)",
                                    bin_id,
                                    bin_info.label,
                                )

                            sorting_service.sort_to_bin(bin_id)
                            logger.info("Sort complete")
                    else:
                        logger.info("No matches found.")
                        # No matches -> overflow bin
                        if args.sort and sorting_service and bin_mapper:
                            sorting_service.sort_to_bin(bin_mapper.overflow_id)
                            logger.info("No match, routed to overflow bin")
                else:
                    logger.error("API Error: %s", result)

            except IOError as e:
                logger.error("Prediction failed: %s", e)

        else:
            logger.error("Failed to capture image.")

    except KeyboardInterrupt:
        logger.info("Stopping app...")

    finally:
        # Cleanup sorting resources
        if sorting_service:
            logger.info("Cleaning up sorting service...")
            sorting_service.cleanup()


if __name__ == "__main__":
    main()
