"""Placeholder sorting service for future sorting mechanism development."""

import logging

from .base_service import AbstractSortingService

logger = logging.getLogger(__name__)


class PlaceholderSortingService(AbstractSortingService):
    """No-op sorting service used before a real mechanism is implemented.

    Logs all sorting actions without performing physical operations.
    Replace with a concrete implementation when the sorting hardware
    is ready.
    """

    def __init__(self, num_bins: int = 6) -> None:
        self._num_bins = num_bins

    def sort_to_bin(self, bin_id: int) -> None:
        """Log a sort action (no-op).

        Args:
            bin_id: Target bin identifier (0-indexed).
        """
        logger.info("PlaceholderSortingService: sort_to_bin(%d) [no-op]", bin_id)

    def get_bin_count(self) -> int:
        """Return the configured number of bins.

        Returns:
            Number of bins.
        """
        return self._num_bins

    def home(self) -> None:
        """Log a home action (no-op)."""
        logger.info("PlaceholderSortingService: home() [no-op]")

    def cleanup(self) -> None:
        """Log a cleanup action (no-op)."""
        logger.info("PlaceholderSortingService: cleanup() [no-op]")
