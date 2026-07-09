"""E2E sorting simulation — exercises the full sorting pipeline with MockGantryClient.

Usage:
    python scripts/e2e_sort_simulation.py

This script validates the complete integration:
    config YAML → Pydantic schemas → BinMapper → MockGantryClient → GantrySortingService
without requiring camera, inference API, or ESP32 hardware.

The fixture in simulate_inference_results() uses part_id/color_id values drawn
from the REAL classifier label namespace (models/part_mapping.json,
models/color_mapping.json) and is cross-referenced against the authoritative
940-entry assignments map in config/bin_layout.yaml (SM-DES-006). Each entry
declares its expected routing outcome (reason + bin id), and main() asserts
the actual outcome against it -- so this is a geometric regression gate, not
just a "no exception raised" smoke test: a future bin_layout.yaml mapping
regression will FAIL the sim.
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

# Structured outcome reason codes -- mirrors sorter_app/main.py's
# REASON_OK / REASON_BELOW_THRESHOLD / REASON_UNMAPPED_PART (S3-03 / O-05 /
# O-06) so this sim's expected-outcome assertions speak the same vocabulary
# as production classification logs. Duplicated here (not imported) to keep
# this script's fixture self-contained, matching its existing style of
# locally redefining CONFIDENCE_THRESHOLD rather than importing it.
REASON_OK = "ok"
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_UNMAPPED_PART = "unmapped_part"


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
    """Simulated inference results mimicking the API response.

    Every part_id below is one of the 94 trained classes in
    models/part_mapping.json, and every color_id is one of the 22 trained
    colors in models/color_mapping.json (real Rebrickable color IDs, named
    via data/raw/rebrickable_20250917/colors.csv). Each entry declares its
    expected_reason / expected_bin_id so main() can assert the routing
    decision, not just "no exception raised".

    NOTE (fixed 2026-07-09, see sw-2-report.md): the previous fixture used
    color_ids 21/24 which do NOT exist in the real 22-color trained
    namespace (stale SM-DES-006 SS6.2 example IDs) -- those cycles could
    never resolve to a real bin under ANY correct mapping and silently fell
    to overflow for the wrong reason. All IDs below are real and verified
    against config/bin_layout.yaml's 940-entry assignments map.
    """
    return [
        {
            # Path: MAPPED -> bin 0 (Yellow). bin_layout.yaml
            # assignments["3004_14"] = 0; color_id 14 = Yellow
            # (SM-DES-006 SS4.2 Bin0 YELLOW).
            "part_id": "3004",
            "color_id": 14,
            "confidence": 0.95,
            "label": "Yellow Brick",
            "expected_reason": REASON_OK,
            "expected_bin_id": 0,
        },
        {
            # Path: MAPPED -> bin 1 (Red). assignments["3004_4"] = 1;
            # color_id 4 = Red. Distinct bin from the entry above, proving
            # the mapping discriminates by color, not just by part.
            "part_id": "3004",
            "color_id": 4,
            "confidence": 0.88,
            "label": "Red Brick",
            "expected_reason": REASON_OK,
            "expected_bin_id": 1,
        },
        {
            # Path: MAPPED -> bin 7 (Grey/Black/White). Uses a DIFFERENT
            # part_id (18746, not 3004) to prove the assignment map
            # generalizes across the full part x color cross product, and
            # exercises bin 7's multi-color aggregation (it pools color_ids
            # 0/15/71/72). assignments["18746_71"] = 7; color_id 71 =
            # Light Bluish Gray.
            "part_id": "18746",
            "color_id": 71,
            "confidence": 0.93,
            "label": "Light Bluish Gray Technic Pin",
            "expected_reason": REASON_OK,
            "expected_bin_id": 7,
        },
        {
            # Path: UNMAPPED -> overflow, reason=unmapped_part. color_id 25
            # = Orange is a real trained color, but bins 4/5 (Orange
            # Large/Small) are intentionally left unassigned in
            # bin_layout.yaml (no size-classification rule exists to split
            # parts into large/small -- see "Bins 4 and 5 are intentionally
            # UNMAPPED" note there), so NO part_id combined with color_id 25
            # resolves to a real bin. A genuine unmapped-assignment probe
            # (unlike the old fixture's fabricated part_id "9999").
            "part_id": "3004",
            "color_id": 25,
            "confidence": 0.90,
            "label": "Orange Brick (unmapped bin)",
            "expected_reason": REASON_UNMAPPED_PART,
            "expected_bin_id": None,
        },
        {
            # Path: LOW CONFIDENCE -> overflow, reason=below_threshold.
            # color_id 2 = Green IS mapped (assignments["3004_2"] = 3), so
            # this deliberately proves the confidence gate is evaluated
            # BEFORE the bin lookup: a part that WOULD resolve to bin 3 at
            # high confidence must still route to overflow because
            # confidence (0.45) < CONFIDENCE_THRESHOLD (0.80).
            "part_id": "3004",
            "color_id": 2,
            "confidence": 0.45,
            "label": "Green Brick (low confidence)",
            "expected_reason": REASON_BELOW_THRESHOLD,
            "expected_bin_id": None,
        },
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
    expectation_failures = 0

    for i, result in enumerate(results, 1):
        part_id = result["part_id"]
        color_id = result["color_id"]
        confidence = result["confidence"]
        label = result["label"]
        expected_reason = result["expected_reason"]
        # None means "overflow, regardless of bin_layout.yaml's configured
        # overflow id" -- resolved against the live BinMapper below rather
        # than hardcoded, so a config change to the overflow bin id doesn't
        # spuriously break this assertion.
        expected_bin_id = (
            bin_mapper.overflow_id
            if result["expected_bin_id"] is None
            else result["expected_bin_id"]
        )

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
                reason = REASON_BELOW_THRESHOLD
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
                    reason = REASON_UNMAPPED_PART
                    logger.info("  No assignment found -> overflow bin %d", bin_id)
                    overflow_count += 1
                else:
                    reason = REASON_OK
                    logger.info("  Assigned to bin %d (%s)", bin_id, bin_info.label)
                    sorted_count += 1

            # Regression gate: the fixture declares what SHOULD happen; if
            # bin_layout.yaml's assignments (or the confidence threshold)
            # drift from what this cycle expects, fail loudly instead of
            # silently passing via overflow-for-the-wrong-reason.
            if reason != expected_reason or bin_id != expected_bin_id:
                logger.error(
                    "  EXPECTATION FAILED: expected reason=%s bin=%d, "
                    "got reason=%s bin=%d",
                    expected_reason,
                    expected_bin_id,
                    reason,
                    bin_id,
                )
                expectation_failures += 1
            else:
                logger.info("  [OK] Matches expected outcome (reason=%s)", reason)

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
    logger.info("  Total cycles:          %d", len(results))
    logger.info("  Sorted to bin:         %d", sorted_count)
    logger.info("  Sent to overflow:      %d", overflow_count)
    logger.info("  Errors:                %d", errors)
    logger.info("  Expectation failures:  %d", expectation_failures)
    passed = errors == 0 and expectation_failures == 0
    logger.info("  Status:                %s", "PASS" if passed else "FAIL")
    logger.info("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
