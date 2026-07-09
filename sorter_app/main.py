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
from sorter_app.domain.schemas import BinLayoutConfig, GantryConfig, SortingConfig
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
    parser.add_argument(
        "--serial-port",
        type=str,
        default=None,
        help=(
            "Override the gantry serial port (e.g. /dev/ttyUSB0 or COM3) "
            "normally read from gantry.yaml. CLI value takes precedence. "
            "Accepted but ignored (no hardware is opened) when --simulate "
            "is also passed."
        ),
    )
    return parser


def resolve_serial_port(configured_port: str, cli_port: str | None) -> str:
    """Resolve the effective gantry serial port.

    Args:
        configured_port: Port loaded from gantry.yaml.
        cli_port: Value of --serial-port, or None if not supplied.

    Returns:
        cli_port if it was supplied (non-empty), else configured_port.
    """
    return cli_port if cli_port else configured_port


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


def resolve_config_path(project_root: str, path: str) -> str:
    """Resolve a (possibly relative) SortingConfig path against project_root.

    Absolute paths are returned unchanged. Relative paths are joined onto
    project_root - this matches the pre-O-02 hardcoded
    ``os.path.join(config_dir, ...)`` / ``os.path.join(data_dir, "captures", ...)``
    behavior exactly, so SortingConfig's defaults resolve to the same
    paths main.py used before this schema existed.

    Args:
        project_root: Absolute path to the repository root.
        path: A path from SortingConfig (relative or absolute).

    Returns:
        Absolute path.
    """
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(project_root, path))


def load_sorting_config(config_path: str) -> SortingConfig:
    """Load the sorting app configuration from YAML, if present.

    Args:
        config_path: Path to sorting.yaml.

    Returns:
        Validated SortingConfig. If the file does not exist, returns
        ``SortingConfig()`` (all field defaults) so main.py behaves the
        same as before config/sorting.yaml was wired up (O-02).
    """
    if not os.path.exists(config_path):
        logger.info("%s not found; using default SortingConfig", config_path)
        return SortingConfig()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return SortingConfig(**(data.get("sorting") or {}))


def main() -> None:
    """Run the sorter application.

    Parses CLI arguments, loads SortingConfig (O-02) and initializes
    services via DI, captures an image (or uses a provided test image),
    and sends it to the inference API. If --sort is specified, sorts the
    part to the appropriate bin.
    """
    logger.info("Starting Lego Sorter App...")

    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine project paths and load the sorting app config (O-02).
    # Falls back to SortingConfig() defaults (== the old hardcoded paths)
    # when config/sorting.yaml is absent or fields are omitted.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sorting_config = load_sorting_config(
        os.path.join(project_root, "config", "sorting.yaml")
    )

    config_service = ConfigService()
    # sorting_config.api_url is None by default, so this preserves the
    # prior behavior (ConfigService's own LEGO_API_URL/localhost default)
    # unless config/sorting.yaml explicitly sets api_url.
    api_url = sorting_config.api_url or config_service.api_url
    api_client = APIClient(base_url=api_url)

    # Initialize sorting services if sorting is enabled
    sorting_service: GantrySortingService | None = None
    bin_mapper: BinMapper | None = None

    if args.sort:
        try:
            # Load configurations (paths now come from SortingConfig, O-02,
            # instead of hardcoded os.path.join(config_dir, ...) calls).
            gantry_config = load_gantry_config(
                resolve_config_path(project_root, sorting_config.gantry_config)
            )
            bin_layout_config = load_bin_layout_config(
                resolve_config_path(project_root, sorting_config.bin_layout_config)
            )

            # Create bin mapper
            bin_mapper = BinMapper(bin_layout_config)

            # CLI --serial-port takes precedence over gantry.yaml (O-01).
            # MockGantryClient never reads gantry_config.serial.port, so
            # this override is a no-op (accepted, not applied) under
            # --simulate.
            resolved_port = resolve_serial_port(
                gantry_config.serial.port, args.serial_port
            )
            if args.serial_port:
                if args.simulate:
                    logger.info(
                        "--serial-port=%s supplied but --simulate is active; "
                        "ignoring for MockGantryClient",
                        args.serial_port,
                    )
                else:
                    logger.info(
                        "Overriding gantry serial port via CLI: %s", resolved_port
                    )
            gantry_config.serial.port = resolved_port

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
        # Default capture output path now comes from SortingConfig (O-02);
        # default value resolves to the same path as the prior hardcoded
        # os.path.join(project_root, "data", "captures", "test_capture.jpg").
        image_path = resolve_config_path(project_root, sorting_config.capture_path)

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
