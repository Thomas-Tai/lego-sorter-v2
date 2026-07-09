"""Shared classification/sort decision path + repeating sorting loop (O-04).

This module holds the single decision path used both by single-shot runs
of ``sorter_app/main.py`` and by ``SortingLoop`` (repeated cycles), so the
loop cannot drift from single-shot behavior - both call the same
``classify_and_sort()`` function.

Also hosts the structured classification reason codes and the
``log_classification_result`` structured log emitter (moved here from
``sorter_app.main`` unchanged - import from this module directly, e.g.
``from sorter_app.services.sorting_loop import log_classification_result``)
plus the ``log_to_db`` hook point (``ClassificationLogger``) that O-09
will wire up to a real ``DatabaseService`` later.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ..domain.bin_mapper import BinMapper
from ..exceptions import BinMappingError, GantryError
from .api_client import APIClient
from .gantry_sorting_service import GantrySortingService
from .part_feed_provider import PartFeedProvider

logger = logging.getLogger("SorterApp")

# Fallback confidence threshold; the live value comes from config/bin_layout.yaml
# via BinMapper.confidence_threshold when a bin mapper is available (S4-01).
CONFIDENCE_THRESHOLD = 0.80

# Structured classification reason codes (S3-03 / O-05 / O-06).
REASON_OK = "ok"
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_UNMAPPED_PART = "unmapped_part"
REASON_NO_MATCH = "no_match"
REASON_API_ERROR = "api_error"
REASON_CLASSIFICATION_FAILED = "classification_failed"


def log_classification_result(
    *,
    image_path: str,
    part_id: str | None,
    color_id: int | None,
    confidence: float | None,
    elapsed_ms: float,
    decision: str,
    reason: str,
) -> None:
    """Emit one structured, machine-parseable classification record.

    This is a single ``key=value`` log line (S3-03) emitted once per
    classification attempt, in addition to the existing human-readable
    log lines - it does not replace them.

    Args:
        image_path: Path to the captured/test image used for this attempt.
        part_id: Identified part id, or None if unavailable.
        color_id: Identified color id, or None if unavailable.
        confidence: Top-match confidence, or None if unavailable.
        elapsed_ms: Wall-clock time spent in the classification API call.
        decision: Outcome bin id as a string, or "none" if not sorted.
        reason: Structured reason code (e.g. "ok", "below_threshold",
            "unmapped_part", "no_match", "api_error",
            "classification_failed").
    """
    logger.info(
        "classification_result image=%s part_id=%s color_id=%s "
        "confidence=%s elapsed_ms=%.1f decision=%s reason=%s",
        os.path.basename(image_path),
        part_id if part_id is not None else "none",
        color_id if color_id is not None else "none",
        f"{confidence:.4f}" if confidence is not None else "none",
        elapsed_ms,
        decision,
        reason,
    )


@dataclass(frozen=True)
class ClassificationRecord:
    """One classification/sort outcome, returned by ``classify_and_sort``.

    Mirrors the fields of ``log_classification_result`` exactly (same
    values, same call), so callers (e.g. ``SortingLoop``, tests, the
    future O-09 ``log_to_db`` implementation) can consume the outcome as
    data instead of re-parsing log lines.

    Attributes:
        image_path: Path to the captured/test image used for this attempt.
        part_id: Identified part id, or None if unavailable.
        color_id: Identified color id, or None if unavailable.
        confidence: Top-match confidence, or None if unavailable.
        elapsed_ms: Wall-clock time spent in the classification API call.
        decision: Outcome bin id as a string, or "none" if not sorted.
        reason: Structured reason code - see the ``REASON_*`` constants.
    """

    image_path: str
    part_id: str | None
    color_id: int | None
    confidence: float | None
    elapsed_ms: float
    decision: str
    reason: str


class ClassificationLogger(ABC):
    """Interface for persisting a classification record ("log_to_db", O-04/O-09).

    ``classify_and_sort`` calls ``log_result`` once per classification
    attempt, after emitting the structured ``log_classification_result``
    line. The real database-backed implementation is O-09 scope (out of
    scope here, and deliberately NOT imported from this module - see
    ``NoOpClassificationLogger``) so this only defines the call site and a
    no-op default, giving ``main.py``/``SortingLoop`` a stable interface to
    depend on today.
    """

    @abstractmethod
    def log_result(self, record: ClassificationRecord) -> None:
        """Persist one classification record.

        Args:
            record: The outcome of one classify_and_sort() call.
        """
        raise NotImplementedError


class NoOpClassificationLogger(ClassificationLogger):
    """No-op ``ClassificationLogger`` - the default ``log_to_db`` hook.

    Does nothing. O-09 will introduce a real implementation backed by
    ``DatabaseService`` (intentionally not imported here per O-04 scope).
    """

    def log_result(self, record: ClassificationRecord) -> None:
        """Discard the record. Intentional no-op (see class docstring)."""
        pass


def classify_and_sort(
    *,
    image_path: str,
    api_client: APIClient,
    sort_enabled: bool,
    sorting_service: GantrySortingService | None,
    bin_mapper: BinMapper | None,
    db_logger: ClassificationLogger | None = None,
) -> ClassificationRecord:
    """Classify one captured image and, if enabled, sort it to a bin.

    This is the single decision path shared by single-shot ``main()`` and
    ``SortingLoop`` (O-04) - extracted verbatim from the pre-O-04
    ``sorter_app.main.main()`` body so the loop cannot drift from
    single-shot behavior. Emits the same human-readable log lines and the
    same structured ``classification_result`` line either way, then
    invokes the ``log_to_db`` hook (``db_logger``, default no-op).

    Args:
        image_path: Path to the image to classify (already captured).
        api_client: Client used to call the inference API.
        sort_enabled: Whether to route the part to a physical bin
            (mirrors the CLI ``--sort`` flag). When False, only
            classification + logging happens.
        sorting_service: GantrySortingService to sort with, or None if
            sorting is not available/enabled.
        bin_mapper: BinMapper for threshold/bin resolution, or None if
            sorting is not available/enabled.
        db_logger: ClassificationLogger hook ("log_to_db"). Defaults to
            ``NoOpClassificationLogger`` when omitted.

    Returns:
        ClassificationRecord describing the outcome (same values as the
        emitted structured log line).
    """
    if db_logger is None:
        db_logger = NoOpClassificationLogger()

    def _emit(
        *,
        part_id: str | None,
        color_id: int | None,
        confidence: float | None,
        elapsed_ms: float,
        decision: str,
        reason: str,
    ) -> ClassificationRecord:
        log_classification_result(
            image_path=image_path,
            part_id=part_id,
            color_id=color_id,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            decision=decision,
            reason=reason,
        )
        record = ClassificationRecord(
            image_path=image_path,
            part_id=part_id,
            color_id=color_id,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            decision=decision,
            reason=reason,
        )
        db_logger.log_result(record)
        return record

    logger.info("Sending to inference API...")
    classify_start = time.perf_counter()
    try:
        result = api_client.predict_from_image(image_path)
        elapsed_ms = (time.perf_counter() - classify_start) * 1000.0

        if result.get("success"):
            matches = result.get("matches", [])
            if matches:
                top_match = matches[0]
                part_id = top_match["part_id"]
                color_id = top_match["color_id"]
                confidence = top_match["confidence"]

                logger.info(
                    "IDENTIFIED: %s (Color: %s)",
                    part_id,
                    color_id,
                )
                logger.info("   Confidence: %s", confidence)
                logger.info("   Source: %s", top_match["source"])

                decision = "none"
                reason = REASON_OK

                # Sort to bin if sorting is enabled
                if sort_enabled and sorting_service and bin_mapper:
                    threshold = getattr(
                        bin_mapper,
                        "confidence_threshold",
                        CONFIDENCE_THRESHOLD,
                    )
                    if confidence < threshold:
                        # Low confidence -> overflow bin (O-05)
                        bin_id = bin_mapper.overflow_id
                        reason = REASON_BELOW_THRESHOLD
                        logger.info(
                            "Low confidence (%.2f < %.2f), routing to overflow bin %d",
                            confidence,
                            threshold,
                            bin_id,
                        )
                    else:
                        # Get bin for part
                        bin_info = bin_mapper.get_bin_for_part(part_id, color_id)
                        bin_id = bin_info.id
                        if bin_id == bin_mapper.overflow_id:
                            # BinMapper silently falls back to
                            # overflow for unmapped parts (O-06).
                            reason = REASON_UNMAPPED_PART
                            logger.info(
                                "No bin mapping for part %s (color %s), "
                                "routing to overflow bin %d",
                                part_id,
                                color_id,
                                bin_id,
                            )
                        else:
                            reason = REASON_OK
                            logger.info(
                                "Routing to bin %d (%s)",
                                bin_id,
                                bin_info.label,
                            )

                    sorting_service.sort_to_bin(bin_id)
                    logger.info("Sort complete")
                    decision = str(bin_id)
                elif confidence < CONFIDENCE_THRESHOLD:
                    reason = REASON_BELOW_THRESHOLD

                return _emit(
                    part_id=part_id,
                    color_id=color_id,
                    confidence=confidence,
                    elapsed_ms=elapsed_ms,
                    decision=decision,
                    reason=reason,
                )
            else:
                logger.info("No matches found.")
                decision = "none"
                # No matches -> overflow bin
                if sort_enabled and sorting_service and bin_mapper:
                    bin_id = bin_mapper.overflow_id
                    sorting_service.sort_to_bin(bin_id)
                    logger.info("No match, routed to overflow bin")
                    decision = str(bin_id)

                return _emit(
                    part_id=None,
                    color_id=None,
                    confidence=None,
                    elapsed_ms=elapsed_ms,
                    decision=decision,
                    reason=REASON_NO_MATCH,
                )
        else:
            logger.error("API Error: %s", result)
            return _emit(
                part_id=None,
                color_id=None,
                confidence=None,
                elapsed_ms=elapsed_ms,
                decision="none",
                reason=REASON_API_ERROR,
            )

    except IOError as e:
        elapsed_ms = (time.perf_counter() - classify_start) * 1000.0
        logger.error("Prediction failed: %s", e)

        # S3-02: classification API failure must still route the
        # part to the overflow bin (same call path as O-06's
        # unmapped-part handling), not just log-and-continue.
        decision = "none"
        if sort_enabled and sorting_service and bin_mapper:
            try:
                bin_id = bin_mapper.overflow_id
                sorting_service.sort_to_bin(bin_id)
                decision = str(bin_id)
                logger.info(
                    "Classification failed, routed to overflow bin %d",
                    bin_id,
                )
            except (GantryError, BinMappingError) as sort_err:
                # A gantry failure during this fallback must not
                # crash the app.
                logger.error(
                    "Overflow routing after classification failure " "also failed: %s",
                    sort_err,
                )

        return _emit(
            part_id=None,
            color_id=None,
            confidence=None,
            elapsed_ms=elapsed_ms,
            decision=decision,
            reason=REASON_CLASSIFICATION_FAILED,
        )


class SortingLoop:
    """Repeats wait_for_part -> capture -> classify -> sort -> log (O-04).

    Runs the target sorting cycle against injected abstractions so real
    hardware can be swapped in later without rewriting the loop:

        - S1 (feeder/conveyor): behind ``feed_provider`` (a
          ``PartFeedProvider`` - see ``sorter_app.services.part_feed_provider``).
          Only ``SimulatedPartFeedProvider`` exists today; a real
          conveyor implementation is out of scope for O-04.
        - S2 (camera/presence): behind ``capture_fn``, a zero-arg callable
          returning the image path to classify for the current cycle.
          Today callers only pass a fixed ``--test-image`` path (no
          per-cycle camera capture exists yet - that is S2 hardware,
          out of scope for O-04).
        - Classification + sort decision: the same ``classify_and_sort``
          used by single-shot ``main()`` - no duplicated logic.
        - ``log_to_db``: behind ``db_logger`` (a ``ClassificationLogger``),
          default no-op (O-09 scope).

    Clean-stop support: pass ``max_cycles`` to ``run()``, call
    ``request_stop()`` from another thread/signal handler to stop before
    the next cycle, or let the ``PartFeedProvider`` itself report
    exhaustion via ``wait_for_part() -> False``. ``feed_provider.stop()``
    is always called on the way out (success, early stop, or exception).
    """

    def __init__(
        self,
        *,
        feed_provider: PartFeedProvider,
        capture_fn: Callable[[], str],
        api_client: APIClient,
        sort_enabled: bool,
        sorting_service: GantrySortingService | None,
        bin_mapper: BinMapper | None,
        db_logger: ClassificationLogger | None = None,
    ) -> None:
        """Initialize the sorting loop.

        Args:
            feed_provider: PartFeedProvider signaling part arrival (S1 seam).
            capture_fn: Zero-arg callable returning the image path to
                classify for the current cycle (S2 seam).
            api_client: Client used to call the inference API.
            sort_enabled: Whether to route parts to physical bins
                (mirrors the CLI ``--sort`` flag).
            sorting_service: GantrySortingService to sort with, or None.
            bin_mapper: BinMapper for threshold/bin resolution, or None.
            db_logger: ClassificationLogger hook ("log_to_db"). Defaults
                to ``NoOpClassificationLogger`` when omitted.
        """
        self._feed_provider = feed_provider
        self._capture_fn = capture_fn
        self._api_client = api_client
        self._sort_enabled = sort_enabled
        self._sorting_service = sorting_service
        self._bin_mapper = bin_mapper
        self._db_logger = db_logger
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request a clean stop before the next cycle begins.

        Safe to call before ``run()`` (the loop then performs zero cycles)
        or during a ``run()`` in progress, e.g. from another thread or a
        signal handler - it takes effect at the start of the next loop
        iteration (a cycle already in progress still completes). A
        ``SortingLoop`` instance is intended for one ``run()`` call (mirrors
        ``PartFeedProvider`` being single-use); ``_stop_requested`` is not
        reset internally, so create a new instance to run again.
        """
        self._stop_requested = True

    def run(self, max_cycles: int | None = None) -> list[ClassificationRecord]:
        """Run the sorting loop.

        Args:
            max_cycles: Maximum number of cycles to run, or None to run
                until the feed provider reports exhaustion or
                ``request_stop()`` is called.

        Returns:
            One ClassificationRecord per completed cycle, in order.
        """
        records: list[ClassificationRecord] = []
        self._feed_provider.start()
        try:
            cycle = 0
            while max_cycles is None or cycle < max_cycles:
                if self._stop_requested:
                    logger.info(
                        "SortingLoop: stop requested, stopping after %d cycle(s)",
                        cycle,
                    )
                    break

                if not self._feed_provider.wait_for_part():
                    logger.info(
                        "SortingLoop: part feed exhausted/stopped after %d cycle(s)",
                        cycle,
                    )
                    break

                image_path = self._capture_fn()
                record = classify_and_sort(
                    image_path=image_path,
                    api_client=self._api_client,
                    sort_enabled=self._sort_enabled,
                    sorting_service=self._sorting_service,
                    bin_mapper=self._bin_mapper,
                    db_logger=self._db_logger,
                )
                records.append(record)
                cycle += 1
        finally:
            self._feed_provider.stop()

        logger.info("SortingLoop finished: %d cycle(s) completed", len(records))
        return records
