"""Tests for GantrySortingService."""

import pytest
from unittest.mock import Mock

from sorter_app.services.gantry_sorting_service import (
    GantrySortingService,
)
from sorter_app.domain.schemas import (
    BinLayoutConfig,
    GridConfig,
    BinEntry,
    OverflowConfig,
)
from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.hardware.mock_gantry import MockGantryClient
from sorter_app.hardware.abstract_gantry import AbstractGantryClient
from sorter_app.exceptions import GantryError, BinMappingError


# Test fixtures
@pytest.fixture
def bin_layout_config() -> BinLayoutConfig:
    """Create a test bin layout configuration."""
    return BinLayoutConfig(
        version=1,
        grid=GridConfig(
            rows=2,
            cols=4,
            bin_width_mm=70.0,
            bin_depth_mm=50.0,
            bin_height_mm=35.0,
            x_spacing_mm=5.0,
            y_spacing_mm=5.0,
            x_offset_mm=55.0,
            y_offset_mm=30.0,
        ),
        bins=[
            BinEntry(id=0, row=0, col=0, label="Yellow"),
            BinEntry(id=1, row=0, col=1, label="Red"),
            BinEntry(id=2, row=0, col=2, label="Blue"),
            BinEntry(id=3, row=0, col=3, label="Green"),
            BinEntry(id=4, row=1, col=0, label="Orange Large"),
            BinEntry(id=5, row=1, col=1, label="Orange Small"),
            BinEntry(id=6, row=1, col=2, label="Magenta"),
            BinEntry(id=7, row=1, col=3, label="Grey/Black"),
        ],
        overflow=OverflowConfig(
            id=8,
            x_mm=280.0,
            y_mm=140.0,
            label="Overflow / Unknown",
        ),
        assignments={},
    )


@pytest.fixture
def bin_mapper(bin_layout_config: BinLayoutConfig) -> BinMapper:
    """Create a test BinMapper."""
    return BinMapper(bin_layout_config)


class TestGantrySortingService:
    """Tests for GantrySortingService."""

    def test_service_initialization(self, bin_mapper: BinMapper) -> None:
        """Test that service initializes correctly."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)

        assert service._gantry is mock_gantry
        assert service._bin_mapper is bin_mapper
        assert service.get_bin_count() == 9  # 8 grid bins + 1 overflow

    def test_sort_to_bin_moves_and_opens_gate(self, bin_mapper: BinMapper) -> None:
        """Test that sort_to_bin calls correct sequence."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)

        # Sort to bin 0
        service.sort_to_bin(0)

        # Verify sequence: move_to, open_gate, close_gate
        # (sleep is hard to verify, but gate operations are checked)
        mock_gantry.move_to.assert_called_once()
        mock_gantry.open_gate.assert_called_once()
        mock_gantry.close_gate.assert_called_once()

        # Verify coordinates for bin 0
        # x = 55 + 0*(70+5) + 35 = 90.0
        # y = 30 + 0*(50+5) + 25 = 55.0
        call_args = mock_gantry.move_to.call_args
        assert call_args[0][0] == 90.0  # x
        assert call_args[0][1] == 55.0  # y

    def test_sort_to_overflow_bin(self, bin_mapper: BinMapper) -> None:
        """Test sorting to overflow bin."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)

        # Sort to overflow bin (id=8)
        service.sort_to_bin(8)

        mock_gantry.move_to.assert_called_once()
        call_args = mock_gantry.move_to.call_args
        assert call_args[0][0] == 280.0  # overflow x
        assert call_args[0][1] == 140.0  # overflow y

    def test_sort_to_invalid_bin_raises(self, bin_mapper: BinMapper) -> None:
        """Test that sorting to invalid bin raises error."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)

        # Try to sort to non-existent bin
        with pytest.raises(BinMappingError, match="Unknown bin_id"):
            service.sort_to_bin(99)

    def test_home_calls_gantry_home(self, bin_mapper: BinMapper) -> None:
        """Test that home() calls gantry.home()."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)
        service.home()

        mock_gantry.home.assert_called_once()

    def test_cleanup_moves_to_origin_and_disconnects(
        self, bin_mapper: BinMapper
    ) -> None:
        """Test that cleanup returns to origin and disconnects."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)
        service.cleanup()

        mock_gantry.move_to.assert_called_once_with(0.0, 0.0)
        mock_gantry.disconnect.assert_called_once()

    def test_cleanup_handles_gantry_error(self, bin_mapper: BinMapper) -> None:
        """Test that cleanup handles GantryError gracefully."""
        mock_gantry = Mock(spec=AbstractGantryClient)
        mock_gantry.move_to.side_effect = GantryError("Test error")

        service = GantrySortingService(mock_gantry, bin_mapper)
        # Should not raise
        service.cleanup()

        # disconnect should still be called even after move_to error
        mock_gantry.disconnect.assert_called_once()

    def test_get_bin_count(self, bin_mapper: BinMapper) -> None:
        """Test that get_bin_count returns correct count."""
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)

        assert service.get_bin_count() == 9  # 8 grid + 1 overflow

    def test_sort_sequence_no_extra_sleep(self, bin_mapper: BinMapper) -> None:
        """Test that sort sequence does not add Python-side sleep.

        Gate hold time (300ms) is handled by firmware before !GATE_DONE,
        so no additional time.sleep is needed in the service layer.
        """
        mock_gantry = Mock(spec=AbstractGantryClient)

        service = GantrySortingService(mock_gantry, bin_mapper)
        service.sort_to_bin(0)

        # Verify the sequence: move_to -> open_gate -> close_gate (no sleep)
        calls = [c[0] for c in mock_gantry.method_calls]
        assert calls == ["move_to", "open_gate", "close_gate"]


class TestGantrySortingServiceIntegration:
    """Integration tests with MockGantryClient."""

    def test_full_sort_cycle(self, bin_mapper: BinMapper) -> None:
        """Test a complete sort cycle with MockGantryClient."""
        from sorter_app.domain.schemas import (
            GantryConfig,
            GantrySerialConfig,
            GantryMotionConfig,
            GantryServoConfig,
            GantrySimulationConfig,
        )

        config = GantryConfig(
            serial=GantrySerialConfig(
                port="/dev/ttyUSB0",
                baud_rate=115200,
                timeout_s=5.0,
                retry_count=2,
            ),
            motion=GantryMotionConfig(
                x_max_mm=355.0,
                y_max_mm=160.0,
                feed_rate_mm_min=3000,
                homing_required=True,
            ),
            servo=GantryServoConfig(
                open_angle_us=2500,
                close_angle_us=500,
                gate_hold_ms=300,
            ),
            simulation=GantrySimulationConfig(
                enabled=True,
                move_delay_s=0.01,  # Fast for testing
            ),
        )

        mock_gantry = MockGantryClient(config)
        service = GantrySortingService(mock_gantry, bin_mapper)

        # Connect and home
        mock_gantry.connect()
        mock_gantry.home()

        # Sort to bin 3
        service.sort_to_bin(3)

        # Verify position is at bin 3
        # x = 55 + 3*(70+5) + 35 = 315.0
        # y = 30 + 0*(50+5) + 25 = 55.0
        x, y = mock_gantry.get_position()
        assert x == 315.0
        assert y == 55.0

        # Move back to origin (not using cleanup which disconnects)
        mock_gantry.move_to(0.0, 0.0)

        # Verify returned to origin
        x, y = mock_gantry.get_position()
        assert x == 0.0
        assert y == 0.0

        # Cleanup
        service.cleanup()
