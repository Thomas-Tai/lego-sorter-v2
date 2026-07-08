"""Hardware abstraction layer for LEGO Sorter V2.

This package provides abstract interfaces and concrete implementations
for communicating with physical hardware (ESP32 gantry controller).
"""

from .abstract_gantry import AbstractGantryClient
from .gantry_client import GantryClient
from .mock_gantry import MockGantryClient

__all__ = [
    "AbstractGantryClient",
    "GantryClient",
    "MockGantryClient",
]
