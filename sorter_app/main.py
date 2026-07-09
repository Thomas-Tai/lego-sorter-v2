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
from sorter_app.exceptions import GantryError, BinMappingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SorterApp")

# Fallback confidence threshold; the live value comes from config/bin_layout.yaml
# via BinMapper.confidence_threshold when a bin mapper is available (S4-01).
CONFIDENCE_THRESHOLD = 0.80

# Structured classification reason codes (S3-03 / O-05 / O-06).
REASON_OK = "ok"
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_UNMAPPED_PART = "unmapped_part"
REASON_NO_MATCH = "no_match"
REASON_API_ERROR = "api_error"
REASON_CLASSIFICATION_FAILED = "classification_failed"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the sorter app.

    Kept separate from ``main()`` so tests can exercise argument parsing
    (including precedence rules) without running the full app.

    Returns:
        Configured ArgumentParser with all sorter app CLI options.
    """
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
    return parser


def log_classification_result(
    *,
    image_path: str,
    part_id: str | None,
    color_id: int | None,
    confidence: float | None,
    elapsed_ms: float,
    decision: str,
    reason: str,
) -> None:
    """Emit one structured, machine-parseable classification record.

    This is a single ``key=value`` log line (S3-03) emitted once per
    classification attempt, in addition to the existing human-readable
    log lines - it does not replace them.

    Args:
        image_path: Path to the captured/test image used for this attempt.
        part_id: Identified part id, or None if unavailable.
        color_id: Identified color id, or None if unavailable.
        confidence: Top-match confidence, or None if unavailable.
        elapsed_ms: Wall-clock time spent in the classification API call.
        decision: Outcome bin id as a string, or "none" if not sorted.
        reason: Structured reason code (e.g. "ok", "below_threshold",
            "unmapped_part", "no_match", "api_error",
            "classification_failed").
    """
    logger.info(
        "classification_result image=%s part_id=%s color_id=%s "
        "confidence=%s elapsed_ms=%.1f decision=%s reason=%s",
        os.path.basename(image_path),
        part_id if part_id is not None else "none",
        color_id if color_id is not None else "none",
        f"{confidence:.4f}" if confidence is not None else "none",
        elapsed_ms,
        decision,
        reason,
    )


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

    parser = build_arg_parser()
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
            classify_start = time.perf_counter()
            try:
                result = api_client.predict_from_image(image_path)
                elapsed_ms = (time.perf_counter() - classify_start) * 1000.0

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

                        decision = "none"
                        reason = REASON_OK

                        # Sort to bin if sorting is enabled
                        if args.sort and sorting_service and bin_mapper:
                            threshold = getattr(
                                bin_mapper,
                                "confidence_threshold",
                                CONFIDENCE_THRESHOLD,
                            )
                            if confidence < threshold:
                                # Low confidence -> overflow bin (O-05)
                                bin_id = bin_mapper.overflow_id
                                reason = REASON_BELOW_THRESHOLD
                                logger.info(
                                    "Low confidence (%.2f < %.2f), routing to overflow bin %d",
                                    confidence,
                                    threshold,
                                    bin_id,
                                )
                            else:
                                # Get bin for part
                                bin_info = bin_mapper.get_bin_for_part(
                                    part_id, color_id
                                )
                                bin_id = bin_info.id
                                if bin_id == bin_mapper.overflow_id:
                                    # BinMapper silently falls back to
                                    # overflow for unmapped parts (O-06).
                                    reason = REASON_UNMAPPED_PART
                                    logger.info(
                                        "No bin mapping for part %s (color %s), "
                                        "routing to overflow bin %d",
                                        part_id,
                                        color_id,
                                        bin_id,
                                    )
                                else:
                                    reason = REASON_OK
                                    logger.info(
                                        "Routing to bin %d (%s)",
                                        bin_id,
                                        bin_info.label,
                                    )

                            sorting_service.sort_to_bin(bin_id)
                            logger.info("Sort complete")
                            decision = str(bin_id)
                        elif confidence < CONFIDENCE_THRESHOLD:
                            reason = REASON_BELOW_THRESHOLD

                        log_classification_result(
                            image_path=image_path,
                            part_id=part_id,
                            color_id=color_id,
                            confidence=confidence,
                            elapsed_ms=elapsed_ms,
                            decision=decision,
                            reason=reason,
                        )
                    else:
                        logger.info("No matches found.")
                        decision = "none"
                        # No matches -> overflow bin
                        if args.sort and sorting_service and bin_mapper:
                            bin_id = bin_mapper.overflow_id
                            sorting_service.sort_to_bin(bin_id)
                            logger.info("No match, routed to overflow bin")
                            decision = str(bin_id)

                        log_classification_result(
                            image_path=image_path,
                            part_id=None,
                            color_id=None,
                            confidence=None,
                            elapsed_ms=elapsed_ms,
                            decision=decision,
                            reason=REASON_NO_MATCH,
                        )
                else:
                    logger.error("API Error: %s", result)
                    log_classification_result(
                        image_path=image_path,
                        part_id=None,
                        color_id=None,
                        confidence=None,
                        elapsed_ms=elapsed_ms,
                        decision="none",
                        reason=REASON_API_ERROR,
                    )

            except IOError as e:
                elapsed_ms = (time.perf_counter() - classify_start) * 1000.0
                logger.error("Prediction failed: %s", e)

                # S3-02: classification API failure must still route the
                # part to the overflow bin (same call path as O-06's
                # unmapped-part handling), not just log-and-continue.
                decision = "none"
                if args.sort and sorting_service and bin_mapper:
                    try:
                        bin_id = bin_mapper.overflow_id
                        sorting_service.sort_to_bin(bin_id)
                        decision = str(bin_id)
                        logger.info(
                            "Classification failed, routed to overflow bin %d",
                            bin_id,
                        )
                    except (GantryError, BinMappingError) as sort_err:
                        # A gantry failure during this fallback must not
                        # crash the app.
                        logger.error(
                            "Overflow routing after classification failure "
                            "also failed: %s",
                            sort_err,
                        )

                log_classification_result(
                    image_path=image_path,
                    part_id=None,
                    color_id=None,
                    confidence=None,
                    elapsed_ms=elapsed_ms,
                    decision=decision,
                    reason=REASON_CLASSIFICATION_FAILED,
                )

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
