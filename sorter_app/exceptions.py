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
    and ESP32 firmware errors. Serves as the catch-all base for the
    more specific GantryTimeoutError and GantryProtocolError below,
    so existing `except GantryError` handlers continue to catch both.
    """

    pass


class GantryTimeoutError(GantryError):
    """Raised when a gantry/UART operation times out.

    This includes serial read timeouts (no bytes received before the
    port timeout elapses) and notification wait timeouts (an expected
    ESP32 notification, e.g. !READY, !HOMED, !MOVE_DONE, does not
    arrive within the configured timeout window).
    """

    pass


class GantryProtocolError(GantryError):
    """Raised when a gantry/UART response violates the wire protocol.

    This includes malformed or unparseable responses (e.g. a
    position response that does not match the expected "X:.. Y:.."
    format), unrecognized response tokens, and firmware error (NACK)
    lines that cannot be parsed into a known error code.
    """

    pass


class BinMappingError(LegoSorterError):
    """Raised when bin mapping fails.

    This includes invalid bin IDs, configuration errors, or
    coordinate validation failures.
    """

    pass
