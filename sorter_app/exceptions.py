"""Custom exception classes for the Lego Sorter application."""


class LegoSorterError(Exception):
    """Base exception for all Lego Sorter errors."""

    pass


class CameraError(LegoSorterError):
    """Raised when camera operations fail."""

    pass


class HardwareError(LegoSorterError):
    """Raised when hardware operations fail."""

    pass
