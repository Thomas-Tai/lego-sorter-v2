"""Integration tests wiring the real production sorting stack together (O-10).

Unlike tests/test_sorting_loop.py (unit-level: Mock sorting service, minimal
in-memory BinLayoutConfig), these tests wire the REAL production pieces end
to end:

    config/gantry.yaml + config/bin_layout.yaml
        (loaded via the same sorter_app.main loaders production uses)
        -> BinMapper (real, production assignments)
        -> MockGantryClient (real simulation-mode hardware client)
        -> GantrySortingService (real)
        -> classify_and_sort / SortingLoop (real shared decision path)

Only the APIClient is stubbed (scripted predict_from_image responses) - no
network, no real serial hardware.

Covers:
    1. High-confidence part mapped in config/bin_layout.yaml assignments
       -> routed to its exact production bin id, reason "ok".
    2. Confidence below the production threshold (0.80 from bin_layout.yaml)
       -> overflow bin, reason below_threshold; threshold value asserted to
       come from the config, and 0.80 exactly is NOT below (strict <).
    3. Part absent from assignments -> overflow, reason unmapped_part.
    4. API success=False -> reason api_error (no sort); API IOError ->
       overflow routing still happens (S3-02 fallback), reason
       classification_failed.
    5. Gantry timeout during the classification-failed overflow fallback
       -> no crash, reason classification_failed, decision "none".
    6. SortingLoop over 4 mixed parts via SimulatedPartFeedProvider -> one
       ClassificationRecord per cycle, in order; feed provider stopped
       afterwards; a record-collecting ClassificationLogger ("log_to_db"
       hook) receives every record.
    7. GantrySortingService.cleanup() runs without error after sorting.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from sorter_app.domain.bin_mapper import BinMapper
from sorter_app.domain.schemas import BinLayoutConfig, GantryConfig
from sorter_app.exceptions import GantryError, GantryTimeoutError
from sorter_app.hardware import MockGantryClient
from sorter_app.main import load_bin_layout_config, load_gantry_config
from sorter_app.services.api_client import APIClient
from sorter_app.services.gantry_sorting_service import GantrySortingService
from sorter_app.services.part_feed_provider import SimulatedPartFeedProvider
from sorter_app.services.sorting_loop import (
    REASON_API_ERROR,
    REASON_BELOW_THRESHOLD,
    REASON_CLASSIFICATION_FAILED,
    REASON_OK,
    REASON_UNMAPPED_PART,
    ClassificationLogger,
    ClassificationRecord,
    SortingLoop,
    classify_and_sort,
)

pytestmark = pytest.mark.integration

# Real production config files - the same ones sorter_app/main.py loads.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GANTRY_CONFIG_PATH = PROJECT_ROOT / "config" / "gantry.yaml"
BIN_LAYOUT_CONFIG_PATH = PROJECT_ROOT / "config" / "bin_layout.yaml"

# Production values from config/bin_layout.yaml (asserted, not assumed, in
# TestProductionConfigWiring below).
OVERFLOW_BIN_ID = 8
OVERFLOW_COORDS = (280.0, 140.0)
PRODUCTION_THRESHOLD = 0.80

IMAGE_PATH = "integration.jpg"


# ---------------------------------------------------------------------------
# Test doubles at the system boundary (APIClient is the ONLY stub; everything
# downstream is the real production implementation).
# ---------------------------------------------------------------------------


class ScriptedAPIClient(APIClient):
    """APIClient whose predict_from_image returns scripted responses.

    Each call consumes the next script entry; an Exception entry is raised
    instead of returned (simulating the IOError the real client raises on
    network failure). Records every image path it was called with.
    """

    def __init__(self, script: Sequence[dict[str, Any] | Exception]) -> None:
        super().__init__(base_url="http://integration-test.invalid")
        self._script = list(script)
        self.calls: list[str] = []

    def predict_from_image(self, image_path: str) -> dict[str, Any]:
        self.calls.append(image_path)
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class RecordingClassificationLogger(ClassificationLogger):
    """Record-collecting ClassificationLogger (the "log_to_db" hook, O-09)."""

    def __init__(self) -> None:
        self.records: list[ClassificationRecord] = []

    def log_result(self, record: ClassificationRecord) -> None:
        self.records.append(record)


class TimeoutDuringSortService(GantrySortingService):
    """Real GantrySortingService whose sort_to_bin always times out.

    Subclass (not a Mock of the subject) so everything except the injected
    failure stays on the production code path.
    """

    def sort_to_bin(self, bin_id: int) -> None:
        raise GantryTimeoutError("simulated !MOVE_DONE timeout")


def _match(part_id: str, color_id: int, confidence: float) -> dict[str, Any]:
    """Build a successful single-match API response."""
    return {
        "success": True,
        "matches": [
            {
                "part_id": part_id,
                "color_id": color_id,
                "confidence": confidence,
                "source": "integration-test",
            }
        ],
    }


def _api_failure() -> dict[str, Any]:
    """Build an API-level failure response (success=False)."""
    return {"success": False, "error": "internal server error"}


# ---------------------------------------------------------------------------
# Fixtures - real production configs and real components
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gantry_config() -> GantryConfig:
    """The REAL production config/gantry.yaml via the production loader."""
    return load_gantry_config(str(GANTRY_CONFIG_PATH))


@pytest.fixture(scope="module")
def bin_layout_config() -> BinLayoutConfig:
    """The REAL production config/bin_layout.yaml via the production loader."""
    return load_bin_layout_config(str(BIN_LAYOUT_CONFIG_PATH))


@pytest.fixture(scope="module")
def bin_mapper(bin_layout_config: BinLayoutConfig) -> BinMapper:
    """Real BinMapper over the production layout (stateless, module-scoped)."""
    return BinMapper(bin_layout_config)


@pytest.fixture
def gantry_client(gantry_config: GantryConfig) -> Iterator[MockGantryClient]:
    """Real MockGantryClient, connected and homed (fresh per test)."""
    client = MockGantryClient(gantry_config)
    client.connect()
    client.home()
    yield client
    client.disconnect()


@pytest.fixture
def sorting_service(
    gantry_client: MockGantryClient, bin_mapper: BinMapper
) -> GantrySortingService:
    """Real GantrySortingService over the real mock gantry + real mapper."""
    return GantrySortingService(gantry_client, bin_mapper)


# ---------------------------------------------------------------------------
# Production config wiring sanity (the values the tests below rely on)
# ---------------------------------------------------------------------------


class TestProductionConfigWiring:
    def test_production_bin_layout_wires_nine_bins(self, bin_mapper: BinMapper) -> None:
        """2x4 grid + overflow from the real YAML = 9 bins, overflow id 8."""
        assert bin_mapper.grid_bin_count == 8
        assert bin_mapper.total_bins == 9
        assert bin_mapper.overflow_id == OVERFLOW_BIN_ID
        assert bin_mapper.get_bin_coordinates(OVERFLOW_BIN_ID) == OVERFLOW_COORDS

    def test_production_grid_formula_places_bin_0_at_90_55(
        self, bin_mapper: BinMapper
    ) -> None:
        """SM-DES-006 grid formula on production offsets: bin 0 center."""
        assert bin_mapper.get_bin_coordinates(0) == (90.0, 55.0)
        assert bin_mapper.get_bin_label(0) == "Yellow"


# ---------------------------------------------------------------------------
# 1. High-confidence mapped part -> its exact production bin id (reason ok)
# ---------------------------------------------------------------------------


class TestMappedPartRouting:
    @pytest.mark.parametrize(
        ("part_id", "color_id", "expected_bin_id"),
        [
            ("3004", 14, 0),  # Yellow -> Bin 0 (Yellow)
            ("3701", 4, 1),  # Red -> Bin 1 (Red)
            ("32524", 322, 2),  # Medium Azure -> Bin 2 (Blue)
            ("6124", 2, 3),  # Green -> Bin 3 (Green)
            ("87079", 26, 6),  # Magenta -> Bin 6 (Magenta)
            ("2431", 71, 7),  # Light Bluish Gray -> Bin 7 (Grey/Black)
        ],
    )
    def test_high_confidence_mapped_part_routes_to_production_bin(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
        gantry_client: MockGantryClient,
        part_id: str,
        color_id: int,
        expected_bin_id: int,
    ) -> None:
        """Real assignments key f"{part_id}_{color_id}" dictates the bin."""
        api_client = ScriptedAPIClient([_match(part_id, color_id, 0.95)])

        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_OK
        assert record.decision == str(expected_bin_id)
        assert record.part_id == part_id
        assert record.color_id == color_id
        # The real MockGantryClient physically "moved" to that bin's center.
        assert gantry_client.get_position() == pytest.approx(
            bin_mapper.get_bin_coordinates(expected_bin_id)
        )


# ---------------------------------------------------------------------------
# 2. Below the production confidence threshold -> overflow (below_threshold)
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_threshold_comes_from_production_config(
        self, bin_mapper: BinMapper, bin_layout_config: BinLayoutConfig
    ) -> None:
        """BinMapper.confidence_threshold is the YAML value, not a constant."""
        assert bin_layout_config.confidence_threshold == pytest.approx(
            PRODUCTION_THRESHOLD
        )
        assert bin_mapper.confidence_threshold == pytest.approx(PRODUCTION_THRESHOLD)

    def test_below_threshold_routes_to_overflow(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
        gantry_client: MockGantryClient,
    ) -> None:
        """A mapped part at 0.79 (< 0.80) goes to overflow, not its bin."""
        api_client = ScriptedAPIClient([_match("3004", 14, 0.79)])

        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_BELOW_THRESHOLD
        assert record.decision == str(OVERFLOW_BIN_ID)
        assert gantry_client.get_position() == pytest.approx(OVERFLOW_COORDS)

    def test_confidence_at_threshold_is_not_below(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
    ) -> None:
        """Exactly 0.80 is NOT below the threshold (strict <) -> reason ok."""
        api_client = ScriptedAPIClient([_match("3004", 14, PRODUCTION_THRESHOLD)])

        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_OK
        assert record.decision == "0"


# ---------------------------------------------------------------------------
# 3. Part absent from assignments -> overflow (unmapped_part)
# ---------------------------------------------------------------------------


class TestUnmappedPartRouting:
    @pytest.mark.parametrize(
        ("part_id", "color_id"),
        [
            ("99999", 14),  # part id not in the production assignments
            ("3004", 999),  # mapped part id, but color has no assignment
        ],
    )
    def test_unmapped_part_routes_to_overflow(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
        gantry_client: MockGantryClient,
        part_id: str,
        color_id: int,
    ) -> None:
        api_client = ScriptedAPIClient([_match(part_id, color_id, 0.95)])

        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_UNMAPPED_PART
        assert record.decision == str(OVERFLOW_BIN_ID)
        assert gantry_client.get_position() == pytest.approx(OVERFLOW_COORDS)


# ---------------------------------------------------------------------------
# 4. API failure paths: success=False (api_error) and IOError (S3-02)
# ---------------------------------------------------------------------------


class TestApiFailurePaths:
    def test_api_error_response_is_not_sorted(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
        gantry_client: MockGantryClient,
    ) -> None:
        """success=False -> reason api_error, no sort, gantry stays home."""
        api_client = ScriptedAPIClient([_api_failure()])

        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_API_ERROR
        assert record.decision == "none"
        assert record.part_id is None
        assert gantry_client.get_position() == (0.0, 0.0)

    def test_ioerror_still_routes_to_overflow(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
        gantry_client: MockGantryClient,
    ) -> None:
        """S3-02: a raised IOError must still route the part to overflow."""
        api_client = ScriptedAPIClient([IOError("connection refused")])

        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_CLASSIFICATION_FAILED
        assert record.decision == str(OVERFLOW_BIN_ID)
        assert gantry_client.get_position() == pytest.approx(OVERFLOW_COORDS)


# ---------------------------------------------------------------------------
# 5. Gantry failure during the classification-failed overflow fallback
# ---------------------------------------------------------------------------


class TestGantryFailureDuringFallback:
    def test_gantry_timeout_during_fallback_does_not_crash(
        self, gantry_client: MockGantryClient, bin_mapper: BinMapper
    ) -> None:
        """GantryTimeoutError while routing the S3-02 fallback is swallowed:
        no crash, reason stays classification_failed, decision "none"."""
        service = TimeoutDuringSortService(gantry_client, bin_mapper)
        api_client = ScriptedAPIClient([IOError("connection refused")])

        # Must complete without raising.
        record = classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=service,
            bin_mapper=bin_mapper,
        )

        assert record.reason == REASON_CLASSIFICATION_FAILED
        assert record.decision == "none"
        # The exception hierarchy the fallback handler relies on: the
        # specific timeout error must be caught as a GantryError.
        assert issubclass(GantryTimeoutError, GantryError)


# ---------------------------------------------------------------------------
# 6. SortingLoop over mixed parts via SimulatedPartFeedProvider
# ---------------------------------------------------------------------------


class TestSortingLoopEndToEnd:
    def test_mixed_parts_produce_ordered_records_and_db_hook_calls(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
    ) -> None:
        """4 cycles (mapped, below-threshold, unmapped, API failure) yield
        one ClassificationRecord each, in order, with correct reasons and
        decisions; the log_to_db hook sees every record; the feed provider
        is stopped afterwards."""
        api_client = ScriptedAPIClient(
            [
                _match("3004", 14, 0.95),  # mapped, confident -> bin 0
                _match("3004", 14, 0.50),  # below threshold -> overflow
                _match("99999", 14, 0.95),  # unmapped -> overflow
                _api_failure(),  # API failure -> not sorted
            ]
        )
        feed_provider = SimulatedPartFeedProvider(part_count=4)
        recorder = RecordingClassificationLogger()
        loop = SortingLoop(
            feed_provider=feed_provider,
            capture_fn=lambda: IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
            db_logger=recorder,
        )

        records = loop.run(max_cycles=4)

        assert [r.reason for r in records] == [
            REASON_OK,
            REASON_BELOW_THRESHOLD,
            REASON_UNMAPPED_PART,
            REASON_API_ERROR,
        ]
        assert [r.decision for r in records] == [
            "0",
            str(OVERFLOW_BIN_ID),
            str(OVERFLOW_BIN_ID),
            "none",
        ]
        # Every cycle captured the same image and hit the API exactly once.
        assert api_client.calls == [IMAGE_PATH] * 4
        # The "log_to_db" hook received every record, in order.
        assert recorder.records == records
        # The feed provider was stopped on the way out of run().
        assert feed_provider.wait_for_part() is False


# ---------------------------------------------------------------------------
# 7. GantrySortingService cleanup after sorting
# ---------------------------------------------------------------------------


class TestServiceCleanup:
    def test_cleanup_after_sorting_completes_without_error(
        self,
        sorting_service: GantrySortingService,
        bin_mapper: BinMapper,
        gantry_client: MockGantryClient,
    ) -> None:
        """After a real sort, cleanup() (gate close + home + disconnect)
        must run without raising and leave the gantry disconnected."""
        api_client = ScriptedAPIClient([_match("3004", 14, 0.95)])
        classify_and_sort(
            image_path=IMAGE_PATH,
            api_client=api_client,
            sort_enabled=True,
            sorting_service=sorting_service,
            bin_mapper=bin_mapper,
        )

        sorting_service.cleanup()  # must not raise

        # cleanup() disconnected the real MockGantryClient - it now rejects
        # further commands (fixture teardown's disconnect() stays safe: the
        # mock's disconnect is idempotent).
        with pytest.raises(GantryError):
            gantry_client.get_position()
