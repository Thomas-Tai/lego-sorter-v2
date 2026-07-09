"""Pydantic configuration schemas for the Lego Sorter application.

DEPRECATED LOCATION (O-02): this module used to define a placeholder
``SortingConfig`` stub (``mechanism_type: "TBD"``) that nothing in the
codebase actually loaded. The real, wired-up ``SortingConfig`` now lives
in ``sorter_app.domain.schemas`` alongside the other root config models
(``GantryConfig``, ``BinLayoutConfig``) that ``sorter_app/main.py`` loads.

This file is kept only so any external reference to
``config.schemas.config_schema.SortingConfig`` keeps resolving; it just
re-exports the canonical class. Prefer importing directly from
``sorter_app.domain.schemas`` in new code.
"""

from sorter_app.domain.schemas import SortingConfig

__all__ = ["SortingConfig"]
