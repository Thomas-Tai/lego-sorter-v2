"""BinMapper - Part-to-coordinate resolution for LEGO Sorter V2.

This module provides the BinMapper class which resolves part_id and color_id
to physical bin coordinates for the sorting mechanism.

Coordinate formula (from SM-DES-006 §7):
    x = x_offset + col * (bin_width + x_spacing) + bin_width / 2
    y = y_offset + row * (bin_depth + y_spacing) + bin_depth / 2
"""

import logging
from typing import Final

from ..exceptions import BinMappingError
from .schemas import BinInfo, BinLayoutConfig

logger = logging.getLogger(__name__)


class BinMapper:
    """Maps part_id + color_id to bin coordinates.

    The BinMapper is stateless - it does not track inventory or modify state.
    It resolves part-to-bin assignments based on the configuration and computes
    physical coordinates for the gantry.

    Attributes:
        config: BinLayoutConfig from bin_layout.yaml.
        overflow_id: ID of the overflow bin.

    Example:
        >>> mapper = BinMapper(config)
        >>> bin_info = mapper.get_bin_for_part("3004", 24)
        >>> print(bin_info.x_mm, bin_info.y_mm)
        90.0 55.0
    """

    # Constants for coordinate validation
    X_MAX_MM: Final[float] = 355.0
    Y_MAX_MM: Final[float] = 160.0

    # Legacy hardcoded threshold, preserved only as a defensive fallback.
    # Historically this value lived as a Python constant (CONFIDENCE_THRESHOLD)
    # in sorter_app/main.py and scripts/e2e_sort_simulation.py. It now lives in
    # config/bin_layout.yaml as BinLayoutConfig.confidence_threshold, which
    # already defaults to this same value, so this fallback should never be
    # exercised in practice (Pydantic guarantees the field is always set).
    DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.80

    def __init__(self, config: BinLayoutConfig) -> None:
        """Initialize BinMapper with configuration.

        Args:
            config: Validated BinLayoutConfig from bin_layout.yaml.

        Raises:
            BinMappingError: If configuration is invalid (e.g., bin coordinates
                exceed gantry limits).
        """
        self._config = config
        self._overflow_id = config.overflow.id

        # Pre-compute bin center coordinates for all grid bins
        self._bin_coords: dict[int, tuple[float, float]] = {}
        self._bin_labels: dict[int, str] = {}

        for bin_entry in config.bins:
            bin_id = bin_entry.id
            row = bin_entry.row
            col = bin_entry.col
            label = bin_entry.label

            # Compute center coordinates using grid formula
            x_mm, y_mm = self._compute_grid_coordinates(row, col)

            # Validate coordinates are within gantry limits
            if x_mm > self.X_MAX_MM or y_mm > self.Y_MAX_MM:
                raise BinMappingError(
                    f"Bin {bin_id} coordinates ({x_mm}, {y_mm}) exceed "
                    f"gantry limits ({self.X_MAX_MM}, {self.Y_MAX_MM})"
                )

            self._bin_coords[bin_id] = (x_mm, y_mm)
            self._bin_labels[bin_id] = label

        # Add overflow bin coordinates
        self._bin_coords[self._overflow_id] = (
            config.overflow.x_mm,
            config.overflow.y_mm,
        )
        self._bin_labels[self._overflow_id] = config.overflow.label

        # Validate overflow bin coordinates
        if config.overflow.x_mm > self.X_MAX_MM or config.overflow.y_mm > self.Y_MAX_MM:
            raise BinMappingError(
                f"Overflow bin coordinates ({config.overflow.x_mm}, "
                f"{config.overflow.y_mm}) exceed gantry limits "
                f"({self.X_MAX_MM}, {self.Y_MAX_MM})"
            )

        logger.info(
            "BinMapper initialized: %d grid bins + 1 overflow = %d total",
            len(config.bins),
            len(self._bin_coords),
        )

    def _compute_grid_coordinates(self, row: int, col: int) -> tuple[float, float]:
        """Compute bin center coordinates from grid position.

        Uses the formula from SM-DES-006 §7:
            x = x_offset + col * (bin_width + x_spacing) + bin_width / 2
            y = y_offset + row * (bin_depth + y_spacing) + bin_depth / 2

        Args:
            row: Row index (0-indexed).
            col: Column index (0-indexed).

        Returns:
            Tuple of (x_mm, y_mm) coordinates.
        """
        grid = self._config.grid

        x_mm = (
            grid.x_offset_mm
            + col * (grid.bin_width_mm + grid.x_spacing_mm)
            + grid.bin_width_mm / 2
        )

        y_mm = (
            grid.y_offset_mm
            + row * (grid.bin_depth_mm + grid.y_spacing_mm)
            + grid.bin_depth_mm / 2
        )

        return (x_mm, y_mm)

    def get_bin_for_part(self, part_id: str, color_id: int) -> BinInfo:
        """Get bin information for a part.

        Resolution order (from SM-SPEC-001 §3.2):
            1. Exact match: assignments["partId:colorId"]
            2. Wildcard color: assignments["partId:*"]
            3. No match: return overflow bin

        Args:
            part_id: LEGO part number (e.g., "3004").
            color_id: LEGO color ID (e.g., 24 for Yellow).

        Returns:
            BinInfo with id, x_mm, y_mm, and label.

        Note:
            This method never raises - unknown parts always return overflow.
        """
        # Try exact match first
        exact_key = f"{part_id}_{color_id}"
        if exact_key in self._config.assignments:
            bin_id = self._config.assignments[exact_key]
            return self._create_bin_info(bin_id)

        # Try wildcard color match
        wildcard_key = f"{part_id}_*"
        if wildcard_key in self._config.assignments:
            bin_id = self._config.assignments[wildcard_key]
            return self._create_bin_info(bin_id)

        # No match - return overflow bin
        logger.debug(
            "No bin mapping for part_id=%s color_id=%s, using overflow",
            part_id,
            color_id,
        )
        return self._create_bin_info(self._overflow_id)

    def get_bin_coordinates(self, bin_id: int) -> tuple[float, float]:
        """Get coordinates for a specific bin ID.

        Args:
            bin_id: Bin identifier (0-indexed).

        Returns:
            Tuple of (x_mm, y_mm) coordinates.

        Raises:
            BinMappingError: If bin_id is not recognized.
        """
        if bin_id not in self._bin_coords:
            raise BinMappingError(f"Unknown bin_id: {bin_id}")

        return self._bin_coords[bin_id]

    def get_bin_label(self, bin_id: int) -> str:
        """Get the label for a specific bin ID.

        Args:
            bin_id: Bin identifier.

        Returns:
            Human-readable bin label.

        Raises:
            BinMappingError: If bin_id is not recognized.
        """
        if bin_id not in self._bin_labels:
            raise BinMappingError(f"Unknown bin_id: {bin_id}")

        return self._bin_labels[bin_id]

    def _create_bin_info(self, bin_id: int) -> BinInfo:
        """Create BinInfo for a bin ID.

        Args:
            bin_id: Bin identifier.

        Returns:
            BinInfo with id, coordinates, and label.

        Raises:
            BinMappingError: If bin_id is not recognized.
        """
        if bin_id not in self._bin_coords:
            raise BinMappingError(f"Assignment references unknown bin_id: {bin_id}")

        x_mm, y_mm = self._bin_coords[bin_id]
        label = self._bin_labels[bin_id]

        return BinInfo(id=bin_id, x_mm=x_mm, y_mm=y_mm, label=label)

    @property
    def confidence_threshold(self) -> float:
        """Return the confidence threshold below which parts route to overflow.

        Sourced from config.confidence_threshold (config/bin_layout.yaml),
        falling back to DEFAULT_CONFIDENCE_THRESHOLD (the old hardcoded
        constant) only if the config value is somehow absent. Pydantic's
        field default already makes that fallback unreachable in practice;
        it exists defensively so behavior is guaranteed unchanged from the
        pre-config hardcoded constant.

        Callers that previously compared confidence against a local
        CONFIDENCE_THRESHOLD constant (e.g. sorter_app/main.py,
        scripts/e2e_sort_simulation.py) can migrate to this property to
        become config-driven; that migration is not done here because
        those files are outside this change's scope.
        """
        return getattr(
            self._config, "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD
        )

    @property
    def overflow_id(self) -> int:
        """Return the overflow bin ID."""
        return self._overflow_id

    @property
    def total_bins(self) -> int:
        """Return total number of bins (grid + overflow)."""
        return len(self._bin_coords)

    @property
    def grid_bin_count(self) -> int:
        """Return number of grid bins (excluding overflow)."""
        return len(self._config.bins)
