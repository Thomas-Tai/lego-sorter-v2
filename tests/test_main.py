"""Tests for sorter_app.main: CLI arg parsing and the classification/sort flow.

Covers the 2026-07-08 evidence-audit gaps:
    S3-02: classification API failure must route the part to overflow.
    S3-03: one structured (key=value) log record per classification.
    O-01:  --serial-port CLI arg, with CLI taking precedence over
           gantry.yaml, ignored gracefully under --simulate.
    O-05:  low-confidence overflow routing carries reason=below_threshold.
    O-06:  unmapped-part overflow routing carries reason=unmapped_part.
"""

import logging
import sys

import pytest

from sorter_app import main as main_module
from sorter_app.exceptions import GantryError
from sorter_app.domain.schemas import (
    BinEntry,
    BinLayoutConfig,
    GantryConfig,
    GantryMotionConfig,
    GantrySerialConfig,
    GantryServoConfig,
    GantrySimulationConfig,
    GridConfig,
    OverflowConfig,
)
from sorter_app.main import (
    REASON_BELOW_THRESHOLD,
    REASON_CLASSIFICATION_FAILED,
    REASON_OK,
    REASON_UNMAPPED_PART,
    build_arg_parser,
    log_classification_result,
    resolve_serial_port,
)

# ---------------------------------------------------------------------------
# O-01: --serial-port CLI argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    """Tests for build_arg_parser (O-01)."""

    def test_serial_port_defaults_to_none(self) -> None:
        args = build_arg_parser().parse_args([])
        assert args.serial_port is None

    def test_serial_port_is_parsed(self) -> None:
        args = build_arg_parser().parse_args(["--serial-port", "COM7"])
        assert args.serial_port == "COM7"

    def test_simulate_and_serial_port_coexist(self) -> None:
        """--simulate must keep working even when --serial-port is given."""
        args = build_arg_parser().parse_args(
            ["--simulate", "--serial-port", "/dev/ttyUSB3"]
        )
        assert args.simulate is True
        assert args.serial_port == "/dev/ttyUSB3"

    def test_existing_flags_unaffected(self) -> None:
        args = build_arg_parser().parse_args(
            ["--sort", "--simulate", "--test-image", "foo.jpg"]
        )
        assert args.sort is True
        assert args.simulate is True
        assert args.test_image == "foo.jpg"


class TestResolveSerialPort:
    """Tests for resolve_serial_port precedence rules (O-01)."""

    def test_cli_value_takes_precedence(self) -> None:
        assert resolve_serial_port("/dev/ttyUSB0", "COM5") == "COM5"

    def test_falls_back_to_configured_port_when_absent(self) -> None:
        assert resolve_serial_port("/dev/ttyUSB0", None) == "/dev/ttyUSB0"

    def test_falls_back_to_configured_port_on_empty_string(self) -> None:
        assert resolve_serial_port("/dev/ttyUSB0", "") == "/dev/ttyUSB0"


# ---------------------------------------------------------------------------
# S3-03: structured classification log record
# ---------------------------------------------------------------------------


class TestLogClassificationResult:
    """Tests for the structured, single-line classification record."""

    def test_emits_one_structured_line_with_all_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="SorterApp"):
            log_classification_result(
                image_path="/tmp/foo/camera_test.jpg",
                part_id="3004",
                color_id=24,
                confidence=0.42,
                elapsed_ms=123.4,
                decision="8",
                reason=REASON_BELOW_THRESHOLD,
            )

        records = [m for m in caplog.messages if m.startswith("classification_result")]
        assert len(records) == 1
        msg = records[0]
        assert "image=camera_test.jpg" in msg
        assert "part_id=3004" in msg
        assert "color_id=24" in msg
        assert "confidence=0.4200" in msg
        assert "elapsed_ms=123.4" in msg
        assert "decision=8" in msg
        assert "reason=below_threshold" in msg

    def test_handles_missing_fields_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="SorterApp"):
            log_classification_result(
                image_path="img.jpg",
                part_id=None,
                color_id=None,
                confidence=None,
                elapsed_ms=5.0,
                decision="none",
                reason=REASON_CLASSIFICATION_FAILED,
            )

        records = [m for m in caplog.messages if m.startswith("classification_result")]
        assert len(records) == 1
        assert "part_id=none" in records[0]
        assert "confidence=none" in records[0]
        assert "reason=classification_failed" in records[0]


# ---------------------------------------------------------------------------
# Fixtures for end-to-end main() flow tests.
#
# These build GantryConfig / BinLayoutConfig objects directly (same pattern
# as tests/test_gantry_sorting_service.py) rather than reading config/*.yaml,
# so the tests stay isolated from concurrent edits to those YAML files.
# ---------------------------------------------------------------------------


@pytest.fixture
def gantry_config() -> GantryConfig:
    return GantryConfig(
        serial=GantrySerialConfig(
            port="/dev/ttyUSB0", baud_rate=115200, timeout_s=5.0, retry_count=2
        ),
        motion=GantryMotionConfig(
            x_max_mm=355.0,
            y_max_mm=160.0,
            feed_rate_mm_min=3000,
            homing_required=True,
        ),
        servo=GantryServoConfig(
            open_angle_us=2500, close_angle_us=500, gate_hold_ms=300
        ),
        simulation=GantrySimulationConfig(enabled=True, move_delay_s=0.0),
    )


@pytest.fixture
def bin_layout_config() -> BinLayoutConfig:
    return BinLayoutConfig(
        version=1,
        grid=GridConfig(
            rows=1,
            cols=1,
            bin_width_mm=70.0,
            bin_depth_mm=50.0,
            bin_height_mm=35.0,
            x_spacing_mm=5.0,
            y_spacing_mm=5.0,
            x_offset_mm=55.0,
            y_offset_mm=30.0,
        ),
        bins=[BinEntry(id=0, row=0, col=0, label="Yellow")],
        overflow=OverflowConfig(
            id=8, x_mm=280.0, y_mm=140.0, label="Overflow / Unknown"
        ),
        assignments={"3004_24": 0},
    )


@pytest.fixture
def patched_config_loaders(monkeypatch, gantry_config, bin_layout_config):
    """Patch the YAML loaders to return in-memory configs (test isolation)."""
    monkeypatch.setattr(main_module, "load_gantry_config", lambda path: gantry_config)
    monkeypatch.setattr(
        main_module, "load_bin_layout_config", lambda path: bin_layout_config
    )


def _run_main_with_argv(monkeypatch, argv: list) -> None:
    monkeypatch.setattr(sys, "argv", ["sorter_app"] + argv)
    main_module.main()


def _make_test_image(tmp_path) -> str:
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")
    return str(image_path)


# ---------------------------------------------------------------------------
# S3-02: classification-failure -> overflow fallback
# ---------------------------------------------------------------------------


class TestClassificationFailureFallback:
    """Tests for S3-02: API failure must route the part to overflow."""

    def test_api_failure_routes_to_overflow_bin(
        self, monkeypatch, tmp_path, caplog, patched_config_loaders
    ) -> None:
        def raise_io_error(self, image_path):
            raise IOError("Connection refused")

        monkeypatch.setattr(main_module.APIClient, "predict_from_image", raise_io_error)

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            _run_main_with_argv(
                monkeypatch,
                ["--simulate", "--sort", "--test-image", image_path],
            )

        messages = caplog.messages
        assert any(
            "Classification failed, routed to overflow bin 8" in m for m in messages
        )

        structured = [m for m in messages if m.startswith("classification_result")]
        assert len(structured) == 1
        assert "reason=classification_failed" in structured[0]
        assert "decision=8" in structured[0]

    def test_api_failure_without_sort_flag_still_emits_structured_record(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """No --sort => no sorting_service => classification_failed with
        decision=none, and the app must not crash (baseline S3-02 guard)."""

        def raise_io_error(self, image_path):
            raise IOError("Connection refused")

        monkeypatch.setattr(main_module.APIClient, "predict_from_image", raise_io_error)

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            _run_main_with_argv(monkeypatch, ["--test-image", image_path])

        structured = [
            m for m in caplog.messages if m.startswith("classification_result")
        ]
        assert len(structured) == 1
        assert "reason=classification_failed" in structured[0]
        assert "decision=none" in structured[0]

    def test_gantry_failure_during_overflow_fallback_does_not_crash(
        self, monkeypatch, tmp_path, caplog, patched_config_loaders
    ) -> None:
        """A gantry failure while routing to overflow after a classification
        failure must be swallowed, not crash the app."""

        def raise_io_error(self, image_path):
            raise IOError("Connection refused")

        monkeypatch.setattr(main_module.APIClient, "predict_from_image", raise_io_error)

        class BrokenSortingService:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def sort_to_bin(self, bin_id: int) -> None:
                raise GantryError("simulated gantry failure")

            def cleanup(self) -> None:
                pass

        monkeypatch.setattr(main_module, "GantrySortingService", BrokenSortingService)

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            # Must complete without raising.
            _run_main_with_argv(
                monkeypatch,
                ["--simulate", "--sort", "--test-image", image_path],
            )

        messages = caplog.messages
        assert any("also failed" in m for m in messages)

        structured = [m for m in messages if m.startswith("classification_result")]
        assert len(structured) == 1
        assert "reason=classification_failed" in structured[0]
        assert "decision=none" in structured[0]


# ---------------------------------------------------------------------------
# O-05 / O-06: structured reason strings on successful classification
# ---------------------------------------------------------------------------


class TestStructuredReasonStrings:
    """Tests for O-05 (below_threshold) and O-06 (unmapped_part)."""

    @staticmethod
    def _predict_with(part_id: str, color_id: int, confidence: float):
        def fake_predict(self, image_path):
            return {
                "success": True,
                "matches": [
                    {
                        "part_id": part_id,
                        "color_id": color_id,
                        "confidence": confidence,
                        "source": "test",
                    }
                ],
            }

        return fake_predict

    def test_low_confidence_reason_is_below_threshold(
        self, monkeypatch, tmp_path, caplog, patched_config_loaders
    ) -> None:
        monkeypatch.setattr(
            main_module.APIClient,
            "predict_from_image",
            self._predict_with("3004", 24, 0.50),  # mapped part, low confidence
        )

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            _run_main_with_argv(
                monkeypatch,
                ["--simulate", "--sort", "--test-image", image_path],
            )

        structured = [
            m for m in caplog.messages if m.startswith("classification_result")
        ]
        assert len(structured) == 1
        assert f"reason={REASON_BELOW_THRESHOLD}" in structured[0]
        assert "decision=8" in structured[0]  # overflow bin id

    def test_unmapped_part_reason_is_unmapped_part(
        self, monkeypatch, tmp_path, caplog, patched_config_loaders
    ) -> None:
        monkeypatch.setattr(
            main_module.APIClient,
            "predict_from_image",
            self._predict_with("9999", 99, 0.95),  # high confidence, unmapped
        )

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            _run_main_with_argv(
                monkeypatch,
                ["--simulate", "--sort", "--test-image", image_path],
            )

        structured = [
            m for m in caplog.messages if m.startswith("classification_result")
        ]
        assert len(structured) == 1
        assert f"reason={REASON_UNMAPPED_PART}" in structured[0]
        assert "decision=8" in structured[0]  # overflow bin id

    def test_mapped_part_high_confidence_reason_is_ok(
        self, monkeypatch, tmp_path, caplog, patched_config_loaders
    ) -> None:
        monkeypatch.setattr(
            main_module.APIClient,
            "predict_from_image",
            self._predict_with("3004", 24, 0.95),  # mapped, high confidence
        )

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            _run_main_with_argv(
                monkeypatch,
                ["--simulate", "--sort", "--test-image", image_path],
            )

        structured = [
            m for m in caplog.messages if m.startswith("classification_result")
        ]
        assert len(structured) == 1
        assert f"reason={REASON_OK}" in structured[0]
        assert "decision=0" in structured[0]  # grid bin id from assignments


# ---------------------------------------------------------------------------
# O-01: --serial-port plumbing end-to-end
# ---------------------------------------------------------------------------


class TestSerialPortPlumbing:
    """Tests that --serial-port reaches the gantry config and is ignored
    gracefully (not crashing) under --simulate."""

    def test_cli_override_applied_and_ignored_gracefully_under_simulate(
        self, monkeypatch, tmp_path, caplog, patched_config_loaders
    ) -> None:
        def raise_io_error(self, image_path):
            raise IOError("no api")

        monkeypatch.setattr(main_module.APIClient, "predict_from_image", raise_io_error)

        captured_configs = []
        original_init = main_module.MockGantryClient.__init__

        def spy_init(self, config):
            captured_configs.append(config)
            original_init(self, config)

        monkeypatch.setattr(main_module.MockGantryClient, "__init__", spy_init)

        image_path = _make_test_image(tmp_path)

        with caplog.at_level(logging.INFO, logger="SorterApp"):
            _run_main_with_argv(
                monkeypatch,
                [
                    "--simulate",
                    "--sort",
                    "--serial-port",
                    "COM9",
                    "--test-image",
                    image_path,
                ],
            )

        assert len(captured_configs) == 1
        # CLI value was applied to the config object passed to the gantry
        # client, even though MockGantryClient never reads it.
        assert captured_configs[0].serial.port == "COM9"
        assert any("ignoring for MockGantryClient" in m for m in caplog.messages)
