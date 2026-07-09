"""Part feed abstraction for the sorting loop (O-04).

Defines the ``PartFeedProvider`` interface that ``SortingLoop`` (see
``sorter_app.services.sorting_loop``) depends on to learn when a new part
is ready to be classified. This is the dependency-injection seam where
the real S1 (feeder/conveyor) and S2 (camera/presence) hardware services
will plug in once built.

NOTE: S1 and S2 hardware do not exist yet. Only ``SimulatedPartFeedProvider``
is implemented here - a synthetic provider used under ``--simulate``/``--loop``
so ``SortingLoop`` can be built and tested against a stable double today.
Adding real hardware later means writing a new ``PartFeedProvider``
implementation (e.g. ``ConveyorPartFeedProvider``) and swapping it in via
dependency injection in ``sorter_app/main.py`` - ``SortingLoop`` itself
does not need to change.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PartFeedProvider(ABC):
    """Abstract interface for feeding parts into the sorting loop.

    Concrete implementations are responsible for starting/stopping the
    physical (or simulated) part feed mechanism (S1: feeder/conveyor) and
    reporting when a part is in position for capture (S2: presence
    sensing). Follows the same house style as ``AbstractGantryClient``
    (``sorter_app.hardware.abstract_gantry``) and the ``Abstract*Service``
    classes in ``sorter_app.services.base_service``: an ABC with blocking
    methods, real timeouts/errors left to concrete implementations.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the feed mechanism (e.g. begin conveyor motion).

        Must be safe to call once before the first ``wait_for_part()``.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_part(self, timeout_s: float | None = None) -> bool:
        """Block until a part is in position for capture, or timeout.

        Args:
            timeout_s: Maximum time to wait in seconds, or None to wait
                indefinitely (implementation-defined; the simulated
                provider never blocks).

        Returns:
            True if a part is ready for capture. False if no part
            arrived (timeout) or the feed is exhausted/stopped - callers
            (``SortingLoop``) treat False as the clean-stop signal.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop the feed mechanism and release resources.

        Must be safe to call multiple times (idempotent), including after
        ``wait_for_part()`` already reported the feed as exhausted.
        """
        raise NotImplementedError


class SimulatedPartFeedProvider(PartFeedProvider):
    """Synthetic ``PartFeedProvider`` for ``--simulate``/``--loop`` (O-04).

    Delivers exactly ``part_count`` synthetic "part ready" signals, then
    reports no more parts on every subsequent call. No real feeder/conveyor
    (S1) or presence sensor (S2) hardware is involved - this is a pure
    in-memory counter, intended purely to exercise ``SortingLoop`` end to
    end without hardware.

    Attributes:
        part_count: Number of synthetic parts to deliver before stopping.
    """

    def __init__(self, part_count: int) -> None:
        """Initialize the simulated provider.

        Args:
            part_count: Number of synthetic "part ready" signals to
                deliver. Must be >= 0 (0 means the feed is exhausted
                immediately - ``wait_for_part()`` always returns False).
        """
        if part_count < 0:
            raise ValueError(f"part_count must be >= 0, got {part_count}")
        self._part_count = part_count
        self._remaining = part_count
        self._started = False
        self._stopped = False

    def start(self) -> None:
        """Mark the simulated feed as running and reset the delivery count."""
        self._started = True
        self._stopped = False
        self._remaining = self._part_count
        logger.info(
            "SimulatedPartFeedProvider started (will deliver %d part(s))",
            self._part_count,
        )

    def wait_for_part(self, timeout_s: float | None = None) -> bool:
        """Deliver one synthetic part, or report the feed is exhausted.

        Returns:
            True and decrements the remaining count if any parts are
            left; False once ``part_count`` parts have been delivered or
            after ``stop()`` has been called.
        """
        if self._stopped or self._remaining <= 0:
            return False
        self._remaining -= 1
        logger.debug(
            "SimulatedPartFeedProvider delivering part (%d remaining after this)",
            self._remaining,
        )
        return True

    def stop(self) -> None:
        """Stop the simulated feed. Idempotent."""
        self._stopped = True
        logger.info("SimulatedPartFeedProvider stopped")
