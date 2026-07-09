"""Tests for sorter_app.services.part_feed_provider (O-04).

Covers PartFeedProvider being a proper ABC (cannot be instantiated
directly) and SimulatedPartFeedProvider's synthetic delivery/exhaustion/
stop/restart behavior.
"""

import pytest

from sorter_app.services.part_feed_provider import (
    PartFeedProvider,
    SimulatedPartFeedProvider,
)


class TestPartFeedProviderIsAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            PartFeedProvider()  # type: ignore[abstract]


class TestSimulatedPartFeedProvider:
    def test_delivers_exactly_part_count_parts(self) -> None:
        provider = SimulatedPartFeedProvider(part_count=3)
        provider.start()

        results = [provider.wait_for_part() for _ in range(3)]

        assert results == [True, True, True]

    def test_reports_exhausted_after_part_count(self) -> None:
        provider = SimulatedPartFeedProvider(part_count=2)
        provider.start()

        for _ in range(2):
            assert provider.wait_for_part() is True

        assert provider.wait_for_part() is False
        assert provider.wait_for_part() is False  # stays exhausted

    def test_zero_part_count_is_immediately_exhausted(self) -> None:
        provider = SimulatedPartFeedProvider(part_count=0)
        provider.start()

        assert provider.wait_for_part() is False

    def test_negative_part_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            SimulatedPartFeedProvider(part_count=-1)

    def test_stop_makes_wait_for_part_return_false(self) -> None:
        provider = SimulatedPartFeedProvider(part_count=5)
        provider.start()
        assert provider.wait_for_part() is True

        provider.stop()

        assert provider.wait_for_part() is False

    def test_stop_is_idempotent(self) -> None:
        provider = SimulatedPartFeedProvider(part_count=1)
        provider.start()
        provider.stop()
        provider.stop()  # must not raise
        assert provider.wait_for_part() is False

    def test_start_resets_delivery_count_for_reuse(self) -> None:
        """start() re-arms the provider - lets a single instance be
        restarted (e.g. in a test loop) rather than requiring a new one."""
        provider = SimulatedPartFeedProvider(part_count=2)
        provider.start()
        assert provider.wait_for_part() is True
        assert provider.wait_for_part() is True
        assert provider.wait_for_part() is False  # exhausted

        provider.start()  # restart

        assert provider.wait_for_part() is True

    def test_wait_for_part_accepts_timeout_argument(self) -> None:
        """Signature compatibility with the PartFeedProvider ABC - the
        simulated provider ignores the timeout but must accept it."""
        provider = SimulatedPartFeedProvider(part_count=1)
        provider.start()

        assert provider.wait_for_part(timeout_s=5.0) is True
