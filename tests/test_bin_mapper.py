"""Tests for BinMapper and domain schemas.

Test cases from Multi_Window_Dev_Prompts.md CP4 checkpoint:
    - test_known_part_returns_assigned_bin
    - test_unknown_part_returns_overflow
    - test_bin_coordinates_match_formula
    - test_overflow_coordinates
    - test_all_bins_within_gantry_bounds
    - test_config_validation_rejects_negative_spacing

Also covers the 2026-07-09 config gap fixes (S4-01/S4-02): the real
config/bin_layout.yaml assignments map (populated from SM-DES-006) and the
confidence_threshold property sourced from config.
"""

import os

import pytest
import yaml

from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.domain.schemas import (
    BinEntry,
    BinLayoutConfig,
    GridConfig,
    OverflowConfig,
)
from sorter_app.exceptions import BinMappingError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_grid_config() -> GridConfig:
    """Create a sample grid configuration matching MVP spec."""
    return GridConfig(
        rows=2,
        cols=4,
        bin_width_mm=70.0,
        bin_depth_mm=50.0,
        bin_height_mm=35.0,
        x_spacing_mm=5.0,
        y_spacing_mm=5.0,
        x_offset_mm=55.0,
        y_offset_mm=30.0,
    )


@pytest.fixture
def sample_bins() -> list[BinEntry]:
    """Create sample bin entries for 2×4 grid."""
    return [
        BinEntry(id=0, row=0, col=0, label="Yellow"),
        BinEntry(id=1, row=0, col=1, label="Red"),
        BinEntry(id=2, row=0, col=2, label="Blue"),
        BinEntry(id=3, row=0, col=3, label="Green"),
        BinEntry(id=4, row=1, col=0, label="Orange Large"),
        BinEntry(id=5, row=1, col=1, label="Orange Small"),
        BinEntry(id=6, row=1, col=2, label="Magenta"),
        BinEntry(id=7, row=1, col=3, label="Grey/Black"),
    ]


@pytest.fixture
def sample_overflow() -> OverflowConfig:
    """Create sample overflow bin config."""
    return OverflowConfig(
        id=8,
        x_mm=280.0,
        y_mm=140.0,
        label="Overflow / Unknown",
    )


@pytest.fixture
def sample_bin_layout_config(
    sample_grid_config: GridConfig,
    sample_bins: list[BinEntry],
    sample_overflow: OverflowConfig,
) -> BinLayoutConfig:
    """Create a complete bin layout configuration with assignments."""
    return BinLayoutConfig(
        version=1,
        grid=sample_grid_config,
        bins=sample_bins,
        overflow=sample_overflow,
        assignments={
            "3004_24": 0,  # Brick 1x2 Yellow → Bin 0
            "3004_21": 1,  # Brick 1x2 Red → Bin 1
            "3004_*": 2,  # Brick 1x2 any color → Bin 2 (wildcard)
        },
    )


@pytest.fixture
def bin_mapper(sample_bin_layout_config: BinLayoutConfig) -> BinMapper:
    """Create a BinMapper instance with sample config."""
    return BinMapper(sample_bin_layout_config)


# =============================================================================
# Test Cases
# =============================================================================


def test_known_part_returns_assigned_bin(bin_mapper: BinMapper) -> None:
    """Test that a known part returns the correctly assigned bin."""
    # Test exact match: "3004_24" → Bin 0 (Yellow)
    bin_info = bin_mapper.get_bin_for_part("3004", 24)
    assert bin_info.id == 0
    assert bin_info.label == "Yellow"

    # Test exact match: "3004_21" → Bin 1 (Red)
    bin_info = bin_mapper.get_bin_for_part("3004", 21)
    assert bin_info.id == 1
    assert bin_info.label == "Red"


def test_unknown_part_returns_overflow(bin_mapper: BinMapper) -> None:
    """Test that unknown parts return the overflow bin."""
    # Part not in assignments → overflow
    bin_info = bin_mapper.get_bin_for_part("9999", 99)
    assert bin_info.id == 8
    assert bin_info.label == "Overflow / Unknown"
    assert bin_info.x_mm == 280.0
    assert bin_info.y_mm == 140.0

    # Color not in assignments (no wildcard for this part) → overflow
    bin_info = bin_mapper.get_bin_for_part("3005", 24)  # Part 3005 not mapped
    assert bin_info.id == 8


def test_wildcard_color_match(bin_mapper: BinMapper) -> None:
    """Test that wildcard color patterns work correctly."""
    # "3004_*" → Bin 2 (Blue) - should match any color not exactly matched
    bin_info = bin_mapper.get_bin_for_part("3004", 99)  # Color 99 not explicitly mapped
    assert bin_info.id == 2
    assert bin_info.label == "Blue"


def test_bin_coordinates_match_formula(bin_mapper: BinMapper) -> None:
    """Test that bin coordinates match the expected formula from SM-DES-006.

    Formula:
        x = x_offset + col * (bin_width + x_spacing) + bin_width / 2
        y = y_offset + row * (bin_depth + y_spacing) + bin_depth / 2

    Expected values:
        bin 0 (row=0, col=0) → (90.0, 55.0)
        bin 3 (row=0, col=3) → (315.0, 55.0)
        bin 4 (row=1, col=0) → (90.0, 110.0)
        bin 7 (row=1, col=3) → (315.0, 110.0)
    """
    # Bin 0: row=0, col=0
    x, y = bin_mapper.get_bin_coordinates(0)
    assert x == pytest.approx(90.0, rel=1e-6)
    assert y == pytest.approx(55.0, rel=1e-6)

    # Bin 3: row=0, col=3
    x, y = bin_mapper.get_bin_coordinates(3)
    assert x == pytest.approx(315.0, rel=1e-6)
    assert y == pytest.approx(55.0, rel=1e-6)

    # Bin 4: row=1, col=0
    x, y = bin_mapper.get_bin_coordinates(4)
    assert x == pytest.approx(90.0, rel=1e-6)
    assert y == pytest.approx(110.0, rel=1e-6)

    # Bin 7: row=1, col=3
    x, y = bin_mapper.get_bin_coordinates(7)
    assert x == pytest.approx(315.0, rel=1e-6)
    assert y == pytest.approx(110.0, rel=1e-6)


def test_overflow_coordinates(bin_mapper: BinMapper) -> None:
    """Test that overflow bin coordinates are correct."""
    x, y = bin_mapper.get_bin_coordinates(8)
    assert x == pytest.approx(280.0, rel=1e-6)
    assert y == pytest.approx(140.0, rel=1e-6)


def test_all_bins_within_gantry_bounds(bin_mapper: BinMapper) -> None:
    """Test that all bin centers are within gantry travel limits.

    Gantry limits: X ≤ 355 mm, Y ≤ 160 mm
    """
    x_max = 355.0
    y_max = 160.0

    for bin_id in range(9):  # bins 0-8
        x, y = bin_mapper.get_bin_coordinates(bin_id)
        assert x <= x_max, f"Bin {bin_id} X={x} exceeds {x_max}"
        assert y <= y_max, f"Bin {bin_id} Y={y} exceeds {y_max}"


def test_config_validation_rejects_negative_spacing() -> None:
    """Test that Pydantic rejects negative spacing values."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        GridConfig(
            rows=2,
            cols=4,
            bin_width_mm=70.0,
            bin_depth_mm=50.0,
            bin_height_mm=35.0,
            x_spacing_mm=-5.0,  # Invalid: negative
            y_spacing_mm=5.0,
            x_offset_mm=55.0,
            y_offset_mm=30.0,
        )


def test_invalid_bin_id_raises_error(bin_mapper: BinMapper) -> None:
    """Test that requesting an invalid bin_id raises BinMappingError."""
    with pytest.raises(BinMappingError):
        bin_mapper.get_bin_coordinates(999)


def test_bin_count_properties(bin_mapper: BinMapper) -> None:
    """Test bin count properties."""
    assert bin_mapper.grid_bin_count == 8  # 2×4 grid
    assert bin_mapper.total_bins == 9  # 8 grid + 1 overflow
    assert bin_mapper.overflow_id == 8


def test_assignment_to_invalid_bin_raises_error(
    sample_grid_config: GridConfig,
    sample_bins: list[BinEntry],
    sample_overflow: OverflowConfig,
) -> None:
    """Test that assignments referencing invalid bin IDs raise error on init."""
    # Assignment references bin 99 which doesn't exist
    bad_config = BinLayoutConfig(
        version=1,
        grid=sample_grid_config,
        bins=sample_bins,
        overflow=sample_overflow,
        assignments={"3004_24": 99},  # Invalid bin ID
    )

    # Getting the bin for this part should raise BinMappingError
    mapper = BinMapper(bad_config)
    with pytest.raises(BinMappingError):
        mapper.get_bin_for_part("3004", 24)


def test_bin_coordinates_exceed_gantry_limits_raises_error(
    sample_bins: list[BinEntry],
    sample_overflow: OverflowConfig,
) -> None:
    """Test that bin coordinates exceeding gantry limits raise error on init."""
    # Create a grid config that would produce coordinates outside limits
    bad_grid = GridConfig(
        rows=2,
        cols=10,  # Too many columns - will exceed X limit
        bin_width_mm=70.0,
        bin_depth_mm=50.0,
        bin_height_mm=35.0,
        x_spacing_mm=5.0,
        y_spacing_mm=5.0,
        x_offset_mm=55.0,
        y_offset_mm=30.0,
    )

    # Create bins for the oversized grid
    large_bins = [
        BinEntry(id=i, row=i // 10, col=i % 10, label=f"Bin {i}") for i in range(20)
    ]

    bad_config = BinLayoutConfig(
        version=1,
        grid=bad_grid,
        bins=large_bins,
        overflow=sample_overflow,
        assignments={},
    )

    with pytest.raises(BinMappingError):
        BinMapper(bad_config)


# =============================================================================
# Real config/bin_layout.yaml assignments (SM-DES-006-derived) + threshold
# =============================================================================


@pytest.fixture
def real_bin_layout_config() -> BinLayoutConfig:
    """Load the actual production config/bin_layout.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "bin_layout.yaml",
    )
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return BinLayoutConfig(**data["bin_layout"])


@pytest.fixture
def real_bin_mapper(real_bin_layout_config: BinLayoutConfig) -> BinMapper:
    return BinMapper(real_bin_layout_config)


def test_real_config_has_nonempty_assignments(
    real_bin_layout_config: BinLayoutConfig,
) -> None:
    """The S4-01 gap (assignments: {}) is closed: real config has mappings."""
    assert len(real_bin_layout_config.assignments) > 0


def test_real_config_yellow_part_resolves_to_bin_0(real_bin_mapper: BinMapper) -> None:
    """Part 3004 (Brick 1x2) in color_id 14 (Yellow) -> Bin 0.

    Traces to SM-DES-006 §2.1 Tray1/Slot0 "Yellow / Bright Yellow pieces"
    and §4.2 "Bin 0 YELLOW". color_id 14 is the real classifier's Yellow
    (models/color_mapping.json, name verified via
    data/raw/rebrickable_20250917/colors.csv), not the doc's stale example
    id (24).
    """
    bin_info = real_bin_mapper.get_bin_for_part("3004", 14)
    assert bin_info.id == 0
    assert bin_info.label == "Yellow"


def test_real_config_blue_part_resolves_to_bin_2(real_bin_mapper: BinMapper) -> None:
    """Part 3004 in color_id 322 (Medium Azure) -> Bin 2.

    Traces to SM-DES-006 §2.1 Tray1/Slot2 "Blue / Medium Azure pieces".
    This is also the one (part, color) pair used by
    scripts/e2e_sort_simulation.py's fixture data that lands in the real
    classifier's 22-color id space.
    """
    bin_info = real_bin_mapper.get_bin_for_part("3004", 322)
    assert bin_info.id == 2
    assert bin_info.label == "Blue"


def test_real_config_orange_part_still_overflows(real_bin_mapper: BinMapper) -> None:
    """Part 3004 in color_id 25 (Orange) is UNVERIFIED/unmapped -> overflow.

    SM-DES-006 splits Orange into Bin 4 (large) / Bin 5 (small) by physical
    part size, but gives no algorithmic rule to classify the 94 trained
    part_ids into large vs small from the doc text alone. Rather than
    invent a size threshold, orange is intentionally left unmapped and
    must safely fall through to overflow (id 8).
    """
    bin_info = real_bin_mapper.get_bin_for_part("3004", 25)
    assert bin_info.id == real_bin_mapper.overflow_id
    assert bin_info.id == 8


def test_real_config_unknown_part_overflows(real_bin_mapper: BinMapper) -> None:
    """A part_id outside the trained 94-class label set still overflows."""
    bin_info = real_bin_mapper.get_bin_for_part("9999", 14)
    assert bin_info.id == real_bin_mapper.overflow_id


def test_real_config_confidence_threshold_matches_legacy_constant(
    real_bin_mapper: BinMapper,
) -> None:
    """BinMapper.confidence_threshold reads 0.80 from bin_layout.yaml.

    0.80 is the same value as the legacy hardcoded CONFIDENCE_THRESHOLD
    constant in sorter_app/main.py / scripts/e2e_sort_simulation.py, so
    moving it into config does not change runtime behavior.
    """
    assert real_bin_mapper.confidence_threshold == pytest.approx(0.80)
