"""Tests for sorter_app.services.sorting_loop (O-04).

Covers:
    - classify_and_sort: the single classify/sort decision path shared by
      single-shot main() and SortingLoop. Verifies each structured reason
      code is produced for the right input (mirrors tests/test_main.py's
      end-to-end coverage, but exercised directly/unit-style here).
    - ClassificationLogger / NoOpClassificationLogger: the log_to_db hook.
    - SortingLoop: runs N cycles, stops cleanly (max_cycles, feed
      exhaustion, and request_stop()), and reuses classify_and_sort
      (no duplicated decision logic).
"""

from unittest.mock import Mock

import pytest

from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.domain.schemas import (
    BinEntry,
    BinLayoutConfig,
    GridConfig,
    OverflowConfig,
)
from sorter_app.services.api_client import APIClient
from sorter_app.services.gantry_sorting_service import GantrySortingService
from sorter_app.services.part_feed_provider import (
    PartFeedProvider,
    SimulatedPartFeedProvider,
)
from sorter_app.services.sorting_loop import (
    REASON_BELOW_THRESHOLD,
    REASON_CLASSIFICATION_FAILED,
    REASON_NO_MATCH,
    REASON_OK,
    REASON_UNMAPPED_PART,
    ClassificationLogger,
    ClassificationRecord,
    NoOpClassificationLogger,
    SortingLoop,
    classify_and_sort,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bin_layout_config() -> BinLayoutConfig:
    return BinLayoutConfig(
        version=1,
        grid=GridConfig(
            rows=1,
            cols=1,
            bin_width_mm=70.0,
            bin_depth_mm=50.0,
            bin_height_mm=35.0,
            x_spacing_mm=5.0,
            y_spacing_mm=5.0,
            x_offset_mm=55.0,
            y_offset_mm=30.0,
        ),
        bins=[BinEntry(id=0, row=0, col=0, label="Yellow")],
        overflow=OverflowConfig(id=8, x_mm=280.0, y_mm=140.0, label="Overflow"),
        assignments={"3004_24": 0},
    )


@pytest.fixture
def bin_mapper(bin_layout_config: BinLayoutConfig) -> BinMapper:
    return BinMapper(bin_layout_config)


def _api_client_with(matches: list | None) -> Mock:
    """Build a Mock APIClient. matches=None simulates an API error response."""
    client = Mock(spec=APIClient)
    if matches is None:
        client.predict_from_image.return_value = {"success": False}
    else:
        client.predict_from_image.return_value = {
            "success": True,
            "matches": matches,
        }
    return client


def _mapped_high_confidence_match() -> list:
    return [{"part_id": "3004", "color_id": 24, "confidence": 0.95, "source": "t"}]


# ---------------------------------------------------------------------------
# classify_and_sort: structured reason routing (mirrors O-05/O-06/S3-02)
# ---------------------------------------------------------------------------


class TestClassifyAndSort:
    def test_ok_reason_when_mapped_and_confident(self, bin_mapper: BinMapper) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_OK
        assert record.decision == "0"
        sorting_service.sort_to_bin.assert_called_once_with(0)

    def test_below_threshold_reason_when_low_confidence(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = _api_client_with(
            [{"part_id": "3004", "color_id": 24, "confidence": 0.5, "source": "t"}]
        )
        sorting_service = Mock(spec=GantrySortingService)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_BELOW_THRESHOLD
        assert record.decision == "8"  # overflow

    def test_unmapped_part_reason(self, bin_mapper: BinMapper) -> None:
        api_client = _api_client_with(
            [{"part_id": "9999", "color_id": 1, "confidence": 0.99, "source": "t"}]
        )
        sorting_service = Mock(spec=GantrySortingService)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_UNMAPPED_PART
        assert record.decision == "8"  # overflow

    def test_no_match_reason(self, bin_mapper: BinMapper) -> None:
        api_client = _api_client_with([])
        sorting_service = Mock(spec=GantrySortingService)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_NO_MATCH
        assert record.decision == "8"

    def test_classification_failed_reason_on_ioerror(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = Mock(spec=APIClient)
        api_client.predict_from_image.side_effect = IOError("connection refused")
        sorting_service = Mock(spec=GantrySortingService)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_CLASSIFICATION_FAILED
        assert record.decision == "8"

    def test_sort_disabled_skips_sorting_service(self, bin_mapper: BinMapper) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=False,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        sorting_service.sort_to_bin.assert_not_called()
        assert record.decision == "none"
        assert record.reason == REASON_OK

    def test_db_logger_hook_called_once_with_matching_record(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)
        db_logger = Mock(spec=ClassificationLogger)

        record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
            db_logger=db_logger,
        )

        db_logger.log_result.assert_called_once_with(record)

    def test_default_db_logger_is_noop_and_does_not_raise(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)

        # No db_logger passed -> defaults to NoOpClassificationLogger.
        classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )


class TestNoOpClassificationLogger:
    def test_log_result_is_a_noop(self) -> None:
        logger_ = NoOpClassificationLogger()
        record = ClassificationRecord(
            image_path="x.jpg",
            part_id=None,
            color_id=None,
            confidence=None,
            elapsed_ms=1.0,
            decision="none",
            reason=REASON_NO_MATCH,
        )
        logger_.log_result(record)  # must not raise


# ---------------------------------------------------------------------------
# SortingLoop
# ---------------------------------------------------------------------------


class TestSortingLoop:
    @staticmethod
    def _feed_provider(n: int) -> PartFeedProvider:
        return SimulatedPartFeedProvider(part_count=n)

    def test_runs_max_cycles_and_returns_one_record_per_cycle(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)
        loop = SortingLoop(
            feed_provider=self._feed_provider(5),
            capture_fn=lambda: "img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        records = loop.run(max_cycles=3)

        assert len(records) == 3
        assert all(r.reason == REASON_OK for r in records)
        assert sorting_service.sort_to_bin.call_count == 3

    def test_stops_cleanly_when_feed_exhausted_before_max_cycles(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)
        loop = SortingLoop(
            feed_provider=self._feed_provider(2),  # fewer parts than max_cycles
            capture_fn=lambda: "img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        records = loop.run(max_cycles=10)

        assert len(records) == 2

    def test_feed_provider_explicitly_stopped_after_run(
        self, bin_mapper: BinMapper
    ) -> None:
        """Even if the feed still has parts left, run() must call
        feed_provider.stop() once max_cycles is reached (clean-stop)."""
        api_client = _api_client_with(_mapped_high_confidence_match())
        feed_provider = self._feed_provider(10)  # more parts than max_cycles
        loop = SortingLoop(
            feed_provider=feed_provider,
            capture_fn=lambda: "img.jpg",
            api_client=api_client,
            sort_enabled=False,
            sorting_service=None,
            bin_mapper=None,
        )

        loop.run(max_cycles=2)

        # 8 parts would still be "available" from the provider's own
        # counter, but explicit stop() must make wait_for_part() False.
        assert feed_provider.wait_for_part() is False

    def test_request_stop_before_run_yields_zero_cycles(
        self, bin_mapper: BinMapper
    ) -> None:
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)
        loop = SortingLoop(
            feed_provider=self._feed_provider(10),
            capture_fn=lambda: "img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        loop.request_stop()
        records = loop.run(max_cycles=10)

        assert records == []
        sorting_service.sort_to_bin.assert_not_called()

    def test_reuses_classify_and_sort_no_duplicated_logic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SortingLoop.run() must call the shared classify_and_sort function
        for each cycle rather than reimplementing the decision logic."""
        import sorter_app.services.sorting_loop as sorting_loop_module

        calls = []

        def spy(**kwargs):
            calls.append(kwargs)
            return ClassificationRecord(
                image_path=kwargs["image_path"],
                part_id=None,
                color_id=None,
                confidence=None,
                elapsed_ms=0.0,
                decision="none",
                reason=REASON_OK,
            )

        monkeypatch.setattr(sorting_loop_module, "classify_and_sort", spy)

        loop = SortingLoop(
            feed_provider=self._feed_provider(3),
            capture_fn=lambda: "img.jpg",
            api_client=Mock(spec=APIClient),
            sort_enabled=False,
            sorting_service=None,
            bin_mapper=None,
        )

        records = loop.run(max_cycles=3)

        assert len(calls) == 3
        assert all(c["image_path"] == "img.jpg" for c in calls)
        assert len(records) == 3

    def test_loop_cycle_matches_direct_classify_and_sort_call(
        self, bin_mapper: BinMapper
    ) -> None:
        """A loop cycle's outcome must equal calling classify_and_sort
        directly with the same inputs - no behavioral drift between the
        single-shot and loop paths."""
        api_client = _api_client_with(_mapped_high_confidence_match())
        sorting_service = Mock(spec=GantrySortingService)

        direct_record = classify_and_sort(
            image_path="img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        loop = SortingLoop(
            feed_provider=self._feed_provider(1),
            capture_fn=lambda: "img.jpg",
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )
        loop_records = loop.run(max_cycles=1)

        assert len(loop_records) == 1
        assert loop_records[0].reason == direct_record.reason
        assert loop_records[0].decision == direct_record.decision
        assert loop_records[0].part_id == direct_record.part_id
