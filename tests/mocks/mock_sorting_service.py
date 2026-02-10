"""Mock sorting service for testing."""

from sorter_app.services.base_service import AbstractSortingService


class MockSortingService(AbstractSortingService):
    """In-memory mock of AbstractSortingService for unit tests.

    Records all calls for assertion in tests.
    """

    def __init__(self, num_bins: int = 6) -> None:
        self._num_bins = num_bins
        self.sort_log: list[int] = []
        self.home_called = False
        self.cleanup_called = False

    def sort_to_bin(self, bin_id: int) -> None:
        """Record a sort action.

        Args:
            bin_id: Target bin identifier.
        """
        self.sort_log.append(bin_id)

    def get_bin_count(self) -> int:
        """Return the configured number of bins.

        Returns:
            Number of bins.
        """
        return self._num_bins

    def home(self) -> None:
        """Record a home action."""
        self.home_called = True

    def cleanup(self) -> None:
        """Record a cleanup action."""
        self.cleanup_called = True
