"""GantrySortingService - Sorting orchestration for LEGO Sorter V2.

This module provides the GantrySortingService class which implements
the AbstractSortingService interface using the gantry hardware.
"""

import logging

from ..domain.bin_mapper import BinMapper
from ..exceptions import GantryError
from ..hardware.abstract_gantry import AbstractGantryClient
from .base_service import AbstractSortingService

logger = logging.getLogger(__name__)


class GantrySortingService(AbstractSortingService):
    """Implementation of AbstractSortingService using gantry hardware.

    This service orchestrates the sort-to-bin operation by coordinating
    the gantry movement and gate operation.

    Attributes:
        _gantry: AbstractGantryClient instance (real or mock).
        _bin_mapper: BinMapper for coordinate resolution.
    """

    def __init__(self, gantry: AbstractGantryClient, bin_mapper: BinMapper) -> None:
        """Initialize GantrySortingService.

        Args:
            gantry: AbstractGantryClient instance (GantryClient or MockGantryClient).
            bin_mapper: BinMapper for resolving bin coordinates.
        """
        self._gantry = gantry
        self._bin_mapper = bin_mapper
        logger.info(
            "GantrySortingService initialized with %d bins",
            bin_mapper.total_bins,
        )

    def sort_to_bin(self, bin_id: int) -> None:
        """Direct a part to the specified sorting bin.

        Sequence:
            1. Resolve bin coordinates via BinMapper
            2. Move gantry to bin position
            3. Open gate
            4. Hold for part to fall
            5. Close gate

        Args:
            bin_id: Target bin identifier (0-indexed).

        Raises:
            GantryError: If any gantry operation fails.
            BinMappingError: If bin_id is invalid.
        """
        # Get bin coordinates
        x_mm, y_mm = self._bin_mapper.get_bin_coordinates(bin_id)
        label = self._bin_mapper.get_bin_label(bin_id)

        logger.info("Sorting to bin %d (%s) at (%.1f, %.1f)", bin_id, label, x_mm, y_mm)

        # Execute sort sequence
        # Note: firmware holds servo open for GATE_HOLD_MS (300ms) before
        # sending !GATE_DONE, so no additional Python-side sleep is needed.
        self._gantry.move_to(x_mm, y_mm)
        self._gantry.open_gate()
        self._gantry.close_gate()

        logger.info("Sort complete: bin %d (%s)", bin_id, label)

    def get_bin_count(self) -> int:
        """Return the number of available sorting bins.

        Returns:
            Total number of bins (grid + overflow).
        """
        return self._bin_mapper.total_bins

    def home(self) -> None:
        """Return the sorting mechanism to its home position."""
        logger.info("Homing gantry")
        self._gantry.home()
        logger.info("Gantry homed")

    def cleanup(self) -> None:
        """Release sorting mechanism resources.

        Returns gantry to home position and disconnects.
        """
        logger.info("Cleaning up GantrySortingService")
        try:
            self._gantry.close_gate()
        except GantryError as e:
            logger.warning("Error closing gate during cleanup: %s", e)

        try:
            self._gantry.move_to(0.0, 0.0)
        except GantryError as e:
            logger.warning("Error returning to home: %s", e)

        try:
            self._gantry.disconnect()
        except GantryError as e:
            logger.warning("Error during disconnect: %s", e)
        logger.info("GantrySortingService cleanup complete")
