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
from sorter_app.services.part_feed_provider import SimulatedPartFeedProvider
from sorter_app.services.sorting_loop import (
    NoOpClassificationLogger,
    SortingLoop,
    classify_and_sort,
)
from sorter_app.domain.schemas import BinLayoutConfig, GantryConfig, SortingConfig
from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.hardware import GantryClient, MockGantryClient
from sorter_app.exceptions import GantryError, BinMappingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SorterApp")

# The structured classification reason codes, log_classification_result,
# ClassificationRecord, ClassificationLogger/NoOpClassificationLogger, and
# classify_and_sort/SortingLoop (the shared single-shot/loop decision path,
# O-04) live in sorter_app.services.sorting_loop - import from there
# directly (tests/test_main.py does).


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
    parser.add_argument(
        "--loop",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run the sorting loop (wait_for_part -> capture -> classify -> "
            "sort -> log, repeated) for N cycles against a simulated "
            "PartFeedProvider instead of a single classification (O-04). "
            "Requires --simulate and --test-image (real S1/S2 hardware and "
            "per-cycle capture do not exist yet; every cycle reuses the "
            "same --test-image). Single-shot behavior (no --loop) is "
            "unaffected and remains the default."
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
    part to the appropriate bin. If --loop N is specified, repeats the
    wait_for_part -> capture -> classify -> sort -> log cycle N times via
    SortingLoop (O-04) instead of running once; single-shot (no --loop)
    behavior is the default and is unchanged.
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
            if args.loop is not None:
                # O-04: repeat the same decision path `args.loop` times
                # against a simulated PartFeedProvider instead of running
                # classify_and_sort once. No real S1 (feeder/conveyor) or
                # S2 (camera/presence) hardware exists yet, so --loop
                # requires --simulate and reuses the single already
                # captured/--test-image image for every cycle.
                if not args.simulate:
                    logger.error(
                        "--loop requires --simulate (no real S1/S2 hardware yet)"
                    )
                    return

                feed_provider = SimulatedPartFeedProvider(part_count=args.loop)
                sorting_loop = SortingLoop(
                    feed_provider=feed_provider,
                    capture_fn=lambda: image_path,
                    api_client=api_client,
                    sort_enabled=args.sort,
                    sorting_service=sorting_service,
                    bin_mapper=bin_mapper,
                    db_logger=NoOpClassificationLogger(),
                )
                records = sorting_loop.run(max_cycles=args.loop)
                logger.info(
                    "Sorting loop complete: %d cycle(s), %d record(s)",
                    args.loop,
                    len(records),
                )
            else:
                # Single-shot (default, unaffected by O-04): same
                # classify_and_sort decision path the loop uses above -
                # extracted to sorter_app.services.sorting_loop so the two
                # call sites cannot drift apart.
                classify_and_sort(
                    image_path=image_path,
                    api_client=api_client,
                    sort_enabled=args.sort,
                    sorting_service=sorting_service,
                    bin_mapper=bin_mapper,
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
