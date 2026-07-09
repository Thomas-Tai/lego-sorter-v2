"""Tests for new config schema fields added by the 2026-07-09 config gap fixes.

Covers:
    - BinLayoutConfig.confidence_threshold (S4-01/S4-02): default, valid
      custom values, and rejection of out-of-range values (must be strictly
      between 0 and 1).
    - GantryConfig.pickup / PickupConfig (S5-29/S5-30): default value,
      custom values, and rejection of invalid (negative) coordinates.
"""

import pytest
from pydantic import ValidationError

from sorter_app.domain.schemas import (
    BinEntry,
    BinLayoutConfig,
    GantryConfig,
    GantryMotionConfig,
    GantrySerialConfig,
    GantryServoConfig,
    GridConfig,
    OverflowConfig,
    PickupConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def grid_config() -> GridConfig:
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
def bins() -> list[BinEntry]:
    return [BinEntry(id=i, row=i // 4, col=i % 4, label=f"Bin {i}") for i in range(8)]


@pytest.fixture
def overflow() -> OverflowConfig:
    return OverflowConfig(id=8, x_mm=280.0, y_mm=140.0, label="Overflow / Unknown")


@pytest.fixture
def gantry_serial() -> GantrySerialConfig:
    return GantrySerialConfig(port="/dev/ttyUSB0")


@pytest.fixture
def gantry_motion() -> GantryMotionConfig:
    return GantryMotionConfig(x_max_mm=355.0, y_max_mm=160.0)


@pytest.fixture
def gantry_servo() -> GantryServoConfig:
    return GantryServoConfig()


# =============================================================================
# BinLayoutConfig.confidence_threshold
# =============================================================================


def test_confidence_threshold_defaults_to_legacy_value(
    grid_config: GridConfig, bins: list[BinEntry], overflow: OverflowConfig
) -> None:
    """Omitting confidence_threshold falls back to 0.80 (legacy constant)."""
    config = BinLayoutConfig(version=1, grid=grid_config, bins=bins, overflow=overflow)
    assert config.confidence_threshold == pytest.approx(0.80)


@pytest.mark.parametrize("value", [0.01, 0.5, 0.80, 0.99])
def test_confidence_threshold_accepts_valid_values(
    grid_config: GridConfig,
    bins: list[BinEntry],
    overflow: OverflowConfig,
    value: float,
) -> None:
    config = BinLayoutConfig(
        version=1,
        grid=grid_config,
        bins=bins,
        overflow=overflow,
        confidence_threshold=value,
    )
    assert config.confidence_threshold == pytest.approx(value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_confidence_threshold_rejects_out_of_range_values(
    grid_config: GridConfig,
    bins: list[BinEntry],
    overflow: OverflowConfig,
    value: float,
) -> None:
    """Threshold must be strictly between 0 and 1 (exclusive both ends)."""
    with pytest.raises(ValidationError):
        BinLayoutConfig(
            version=1,
            grid=grid_config,
            bins=bins,
            overflow=overflow,
            confidence_threshold=value,
        )


# =============================================================================
# GantryConfig.pickup / PickupConfig
# =============================================================================


def test_pickup_defaults_when_omitted(
    gantry_serial: GantrySerialConfig,
    gantry_motion: GantryMotionConfig,
    gantry_servo: GantryServoConfig,
) -> None:
    """GantryConfig built without a `pickup` key gets safe (0.0, 0.0) defaults.

    This is the yaml-less construction pattern used by existing tests
    (tests/test_gantry_client.py, tests/test_gantry_sorting_service.py) —
    it must keep working unchanged after adding the pickup field.
    """
    config = GantryConfig(
        serial=gantry_serial, motion=gantry_motion, servo=gantry_servo
    )
    assert isinstance(config.pickup, PickupConfig)
    assert config.pickup.x_mm == pytest.approx(0.0)
    assert config.pickup.y_mm == pytest.approx(0.0)


def test_pickup_accepts_custom_coordinates(
    gantry_serial: GantrySerialConfig,
    gantry_motion: GantryMotionConfig,
    gantry_servo: GantryServoConfig,
) -> None:
    config = GantryConfig(
        serial=gantry_serial,
        motion=gantry_motion,
        servo=gantry_servo,
        pickup=PickupConfig(x_mm=12.5, y_mm=7.0),
    )
    assert config.pickup.x_mm == pytest.approx(12.5)
    assert config.pickup.y_mm == pytest.approx(7.0)


@pytest.mark.parametrize("field,value", [("x_mm", -1.0), ("y_mm", -1.0)])
def test_pickup_rejects_negative_coordinates(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        PickupConfig(**{field: value})


def test_gantry_yaml_loads_pickup_section() -> None:
    """config/gantry.yaml's new `pickup:` section parses into PickupConfig."""
    import os

    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "gantry.yaml",
    )
    with open(config_path) as f:
        data = yaml.safe_load(f)

    assert "pickup" in data["gantry"]
    config = GantryConfig(**data["gantry"])
    assert config.pickup.x_mm == pytest.approx(0.0)
    assert config.pickup.y_mm == pytest.approx(0.0)


def test_bin_layout_yaml_loads_confidence_threshold() -> None:
    """config/bin_layout.yaml's confidence_threshold parses and matches legacy 0.80."""
    import os

    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "bin_layout.yaml",
    )
    with open(config_path) as f:
        data = yaml.safe_load(f)

    assert data["bin_layout"]["confidence_threshold"] == pytest.approx(0.80)
    config = BinLayoutConfig(**data["bin_layout"])
    assert config.confidence_threshold == pytest.approx(0.80)
