"""Tests for the gantry exception hierarchy in sorter_app.exceptions.

Covers GantryError, GantryTimeoutError, and GantryProtocolError:
class relationships, message propagation, and that existing
`except GantryError` handlers still catch the new subclasses.
"""

import pytest

from sorter_app.exceptions import (
    GantryError,
    GantryProtocolError,
    GantryTimeoutError,
    LegoSorterError,
)


class TestGantryErrorHierarchy:
    """Tests for the GantryError class hierarchy."""

    def test_gantry_error_is_lego_sorter_error(self) -> None:
        """Test that GantryError subclasses LegoSorterError."""
        assert issubclass(GantryError, LegoSorterError)

    def test_gantry_timeout_error_subclasses_gantry_error(self) -> None:
        """Test that GantryTimeoutError subclasses GantryError."""
        assert issubclass(GantryTimeoutError, GantryError)

    def test_gantry_protocol_error_subclasses_gantry_error(self) -> None:
        """Test that GantryProtocolError subclasses GantryError."""
        assert issubclass(GantryProtocolError, GantryError)

    def test_gantry_timeout_and_protocol_are_distinct(self) -> None:
        """Test that the two new subclasses are not related to each other."""
        assert not issubclass(GantryTimeoutError, GantryProtocolError)
        assert not issubclass(GantryProtocolError, GantryTimeoutError)

    @pytest.mark.parametrize(
        "exc_type", [GantryError, GantryTimeoutError, GantryProtocolError]
    )
    def test_all_gantry_errors_subclass_lego_sorter_error(
        self, exc_type: type[Exception]
    ) -> None:
        """Test that every gantry exception ultimately traces to LegoSorterError."""
        assert issubclass(exc_type, LegoSorterError)

    @pytest.mark.parametrize(
        "exc_type", [GantryTimeoutError, GantryProtocolError, GantryError]
    )
    def test_message_is_preserved(self, exc_type: type[Exception]) -> None:
        """Test that the exception message is preserved on the instance."""
        exc = exc_type("something went wrong")
        assert str(exc) == "something went wrong"

    @pytest.mark.parametrize("exc_type", [GantryTimeoutError, GantryProtocolError])
    def test_caught_by_except_gantry_error(self, exc_type: type[Exception]) -> None:
        """Test that existing `except GantryError` handlers still catch these.

        This is the core compatibility guarantee for S5-28: refining the
        raise sites in GantryClient to use the new subclasses must not
        break any code that only knows about the base GantryError.
        """
        caught: Exception | None = None
        try:
            raise exc_type("boom")
        except GantryError as e:
            caught = e

        assert caught is not None
        assert isinstance(caught, exc_type)

    @pytest.mark.parametrize("exc_type", [GantryTimeoutError, GantryProtocolError])
    def test_has_docstring(self, exc_type: type[Exception]) -> None:
        """Test that the new exception classes document their purpose."""
        assert exc_type.__doc__ is not None
        assert len(exc_type.__doc__.strip()) > 0
