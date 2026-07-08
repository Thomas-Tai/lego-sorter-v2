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


class GantryError(LegoSorterError):
    """Raised when gantry/UART communication fails.

    This includes serial timeouts, protocol errors, motion errors,
    and ESP32 firmware errors.
    """

    pass


class BinMappingError(LegoSorterError):
    """Raised when bin mapping fails.

    This includes invalid bin IDs, configuration errors, or
    coordinate validation failures.
    """

    pass
