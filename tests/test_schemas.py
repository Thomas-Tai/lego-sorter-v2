"""Tests for new config schema fields added by the 2026-07-09 config gap fixes.

Covers:
    - BinLayoutConfig.confidence_threshold (S4-01/S4-02): default, valid
      custom values, and rejection of out-of-range values (must be strictly
      between 0 and 1).
    - GantryConfig.pickup / PickupConfig (S5-29/S5-30): default value,
      custom values, and rejection of invalid (negative) coordinates.
    - SortingConfig (O-02): defaults must equal the pre-O-02 hardcoded
      paths, field validation, optional api_url, the
      config.schemas.config_schema re-export shim, plus main.py's
      resolve_config_path()/load_sorting_config() fallback behaviors.
"""

import os
from pathlib import Path

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
    SortingConfig,
)
from sorter_app.main import load_sorting_config, resolve_config_path

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


# =============================================================================
# SortingConfig (O-02)
# =============================================================================


def test_sorting_config_defaults_match_legacy_hardcoded_paths() -> None:
    """SortingConfig() defaults must equal the pre-O-02 hardcoded paths.

    Before O-02, main.py built these paths with os.path.join(config_dir,
    "gantry.yaml") / os.path.join(config_dir, "bin_layout.yaml") /
    os.path.join(project_root, "data", "captures", "test_capture.jpg").
    The schema defaults must resolve to the same locations so a missing
    or empty config/sorting.yaml changes nothing.
    """
    config = SortingConfig()
    assert config.gantry_config == "config/gantry.yaml"
    assert config.bin_layout_config == "config/bin_layout.yaml"
    assert config.capture_path == "data/captures/test_capture.jpg"


def test_sorting_config_api_url_defaults_to_none() -> None:
    """api_url=None (default) means ConfigService's own resolution wins."""
    assert SortingConfig().api_url is None


def test_sorting_config_accepts_custom_values() -> None:
    config = SortingConfig(
        gantry_config="alt/gantry.yaml",
        bin_layout_config="alt/bin_layout.yaml",
        capture_path="alt/captures/shot.jpg",
        api_url="http://legosorter.local:8000",
    )
    assert config.gantry_config == "alt/gantry.yaml"
    assert config.bin_layout_config == "alt/bin_layout.yaml"
    assert config.capture_path == "alt/captures/shot.jpg"
    assert config.api_url == "http://legosorter.local:8000"


@pytest.mark.parametrize(
    "field", ["gantry_config", "bin_layout_config", "capture_path", "api_url"]
)
def test_sorting_config_rejects_empty_string(field: str) -> None:
    """All fields carry min_length=1 - '' must not silently disable a path."""
    with pytest.raises(ValidationError):
        SortingConfig(**{field: ""})


@pytest.mark.parametrize(
    "field", ["gantry_config", "bin_layout_config", "capture_path"]
)
def test_sorting_config_rejects_none_for_required_paths(field: str) -> None:
    """Path fields are non-optional: None is invalid (unlike api_url)."""
    with pytest.raises(ValidationError):
        SortingConfig(**{field: None})  # type: ignore[arg-type]


def test_config_schema_shim_reexports_canonical_class() -> None:
    """config.schemas.config_schema.SortingConfig is the canonical class.

    The old stub location is kept as a pure re-export (O-02); it must be
    the *same object*, not a diverging copy. Imported via importlib so
    mypy does not map the file to a second static module name (config/
    is a namespace package with no __init__.py).
    """
    import importlib

    shim_module = importlib.import_module("config.schemas.config_schema")
    assert shim_module.SortingConfig is SortingConfig


def test_production_sorting_yaml_parses_and_matches_defaults() -> None:
    """config/sorting.yaml ships the same values as the schema defaults.

    api_url is deliberately absent (commented out) in the shipped file,
    so parsing it must produce a config equal to SortingConfig().
    """
    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "sorting.yaml",
    )
    with open(config_path) as f:
        data = yaml.safe_load(f)

    config = SortingConfig(**data["sorting"])
    assert config == SortingConfig()


# =============================================================================
# resolve_config_path / load_sorting_config (O-02, main.py)
# =============================================================================


def test_resolve_config_path_joins_relative_to_project_root() -> None:
    resolved = resolve_config_path("/proj/root", "config/gantry.yaml")
    assert resolved == os.path.normpath(
        os.path.join("/proj/root", "config/gantry.yaml")
    )


def test_resolve_config_path_returns_absolute_path_unchanged(tmp_path: Path) -> None:
    """Absolute paths must ignore project_root entirely."""
    abs_path = str(tmp_path / "elsewhere" / "gantry.yaml")
    resolved = resolve_config_path("/proj/root", abs_path)
    assert resolved == os.path.normpath(abs_path)
    assert "proj" not in resolved


def test_load_sorting_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    """No config/sorting.yaml -> pure defaults (pre-O-02 behavior)."""
    config = load_sorting_config(str(tmp_path / "does_not_exist.yaml"))
    assert config == SortingConfig()


def test_load_sorting_config_empty_file_returns_defaults(tmp_path: Path) -> None:
    """An empty YAML file (safe_load -> None) must not crash: defaults."""
    config_file = tmp_path / "sorting.yaml"
    config_file.write_text("", encoding="utf-8")
    config = load_sorting_config(str(config_file))
    assert config == SortingConfig()


def test_load_sorting_config_missing_sorting_key_returns_defaults(
    tmp_path: Path,
) -> None:
    """A YAML file without a `sorting:` section falls back to defaults."""
    config_file = tmp_path / "sorting.yaml"
    config_file.write_text("unrelated: true\n", encoding="utf-8")
    config = load_sorting_config(str(config_file))
    assert config == SortingConfig()


def test_load_sorting_config_partial_fields_keep_remaining_defaults(
    tmp_path: Path,
) -> None:
    """Setting only api_url must leave every path field at its default."""
    config_file = tmp_path / "sorting.yaml"
    config_file.write_text(
        "sorting:\n  api_url: http://legosorter.local:8000\n", encoding="utf-8"
    )
    config = load_sorting_config(str(config_file))
    assert config.api_url == "http://legosorter.local:8000"
    assert config.gantry_config == "config/gantry.yaml"
    assert config.bin_layout_config == "config/bin_layout.yaml"
    assert config.capture_path == "data/captures/test_capture.jpg"


def test_load_sorting_config_reads_all_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "sorting.yaml"
    config_file.write_text(
        "sorting:\n"
        "  gantry_config: alt/gantry.yaml\n"
        "  bin_layout_config: alt/bin_layout.yaml\n"
        "  capture_path: alt/shot.jpg\n"
        "  api_url: http://pi:9000\n",
        encoding="utf-8",
    )
    config = load_sorting_config(str(config_file))
    assert config.gantry_config == "alt/gantry.yaml"
    assert config.bin_layout_config == "alt/bin_layout.yaml"
    assert config.capture_path == "alt/shot.jpg"
    assert config.api_url == "http://pi:9000"
