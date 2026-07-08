"""E2E sorting simulation — exercises the full sorting pipeline with MockGantryClient.

Usage:
    python scripts/e2e_sort_simulation.py

This script validates the complete integration:
    config YAML → Pydantic schemas → BinMapper → MockGantryClient → GantrySortingService
without requiring camera, inference API, or ESP32 hardware.
"""

import logging
import os
import sys
import time

import yaml

# Ensure sorter_app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sorter_app.domain.schemas import (
    BinLayoutConfig,
    GantryConfig,
    GantrySimulationConfig,
)
from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.hardware.mock_gantry import MockGantryClient
from sorter_app.services.gantry_sorting_service import GantrySortingService
from sorter_app.exceptions import GantryError, BinMappingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("E2E-Simulation")

CONFIDENCE_THRESHOLD = 0.80


def load_configs(config_dir: str) -> tuple[GantryConfig, BinLayoutConfig]:
    """Load and validate YAML configs via Pydantic."""
    with open(os.path.join(config_dir, "gantry.yaml")) as f:
        gantry_data = yaml.safe_load(f)
    with open(os.path.join(config_dir, "bin_layout.yaml")) as f:
        bin_data = yaml.safe_load(f)

    gantry_config = GantryConfig(**gantry_data["gantry"])
    # Force simulation mode with fast delays
    gantry_config.simulation = GantrySimulationConfig(enabled=True, move_delay_s=0.02)

    bin_layout_config = BinLayoutConfig(**bin_data["bin_layout"])
    return gantry_config, bin_layout_config


def simulate_inference_results() -> list[dict]:
    """Simulated inference results mimicking the API response."""
    return [
        {
            "part_id": "3004",
            "color_id": 24,
            "confidence": 0.95,
            "label": "Yellow Brick",
        },
        {"part_id": "3004", "color_id": 21, "confidence": 0.88, "label": "Red Brick"},
        {"part_id": "9999", "color_id": 0, "confidence": 0.92, "label": "Unknown Part"},
        {
            "part_id": "3004",
            "color_id": 24,
            "confidence": 0.45,
            "label": "Low Confidence",
        },
        {"part_id": "3004", "color_id": 322, "confidence": 0.91, "label": "Blue Brick"},
    ]


def main() -> None:
    logger.info("=" * 60)
    logger.info("E2E SORTING SIMULATION — MockGantryClient")
    logger.info("=" * 60)

    # Step 1: Load configs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(project_root, "config")
    logger.info("Loading configs from %s", config_dir)

    gantry_config, bin_layout_config = load_configs(config_dir)
    logger.info("[OK] Configs loaded and validated")

    # Step 2: Create BinMapper
    bin_mapper = BinMapper(bin_layout_config)
    logger.info(
        "[OK] BinMapper created: %d grid bins + overflow (id=%d)",
        bin_mapper.grid_bin_count,
        bin_mapper.overflow_id,
    )

    # Verify bin coordinates
    for bin_id in range(bin_mapper.grid_bin_count):
        x, y = bin_mapper.get_bin_coordinates(bin_id)
        label = bin_mapper.get_bin_label(bin_id)
        logger.info("     Bin %d (%s): (%.1f, %.1f) mm", bin_id, label, x, y)
    ox, oy = bin_mapper.get_bin_coordinates(bin_mapper.overflow_id)
    logger.info(
        "     Overflow (id=%d): (%.1f, %.1f) mm", bin_mapper.overflow_id, ox, oy
    )

    # Step 3: Create MockGantryClient
    gantry = MockGantryClient(gantry_config)
    logger.info("[OK] MockGantryClient created")

    # Step 4: Connect + Home
    gantry.connect()
    logger.info("[OK] Gantry connected")
    gantry.home()
    pos = gantry.get_position()
    logger.info("[OK] Homed at (%.1f, %.1f)", pos[0], pos[1])

    # Step 5: Create GantrySortingService
    sorting_service = GantrySortingService(gantry, bin_mapper)
    logger.info(
        "[OK] GantrySortingService created (bin_count=%d)",
        sorting_service.get_bin_count(),
    )

    # Step 6: Simulate sort cycles
    logger.info("")
    logger.info("-" * 60)
    logger.info("SIMULATING %d SORT CYCLES", len(simulate_inference_results()))
    logger.info("-" * 60)

    results = simulate_inference_results()
    sorted_count = 0
    overflow_count = 0
    errors = 0

    for i, result in enumerate(results, 1):
        part_id = result["part_id"]
        color_id = result["color_id"]
        confidence = result["confidence"]
        label = result["label"]

        logger.info("")
        logger.info(
            "--- Cycle %d: %s (part=%s, color=%d, conf=%.2f) ---",
            i,
            label,
            part_id,
            color_id,
            confidence,
        )

        try:
            if confidence < CONFIDENCE_THRESHOLD:
                bin_id = bin_mapper.overflow_id
                logger.info(
                    "  Low confidence (%.2f < %.2f) -> overflow bin %d",
                    confidence,
                    CONFIDENCE_THRESHOLD,
                    bin_id,
                )
                overflow_count += 1
            else:
                bin_info = bin_mapper.get_bin_for_part(part_id, color_id)
                bin_id = bin_info.id
                if bin_id == bin_mapper.overflow_id:
                    logger.info("  No assignment found -> overflow bin %d", bin_id)
                    overflow_count += 1
                else:
                    logger.info("  Assigned to bin %d (%s)", bin_id, bin_info.label)
                    sorted_count += 1

            t0 = time.perf_counter()
            sorting_service.sort_to_bin(bin_id)
            elapsed = time.perf_counter() - t0

            pos = gantry.get_position()
            logger.info(
                "  Sort complete in %.0f ms, gantry at (%.1f, %.1f)",
                elapsed * 1000,
                pos[0],
                pos[1],
            )

        except (GantryError, BinMappingError) as e:
            logger.error("  SORT FAILED: %s", e)
            errors += 1

    # Step 7: Cleanup
    logger.info("")
    logger.info("-" * 60)
    logger.info("CLEANUP")
    logger.info("-" * 60)
    sorting_service.home()
    pos = gantry.get_position()
    logger.info("[OK] Returned to home (%.1f, %.1f)", pos[0], pos[1])
    sorting_service.cleanup()
    logger.info("[OK] Service cleaned up")

    # Step 8: Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("E2E SIMULATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Total cycles:     %d", len(results))
    logger.info("  Sorted to bin:    %d", sorted_count)
    logger.info("  Sent to overflow: %d", overflow_count)
    logger.info("  Errors:           %d", errors)
    logger.info("  Status:           %s", "PASS" if errors == 0 else "FAIL")
    logger.info("=" * 60)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
