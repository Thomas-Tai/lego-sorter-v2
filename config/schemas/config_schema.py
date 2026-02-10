"""Pydantic configuration schemas for the Lego Sorter application."""

from pydantic import BaseModel


class SortingConfig(BaseModel):
    """Validation schema for config/sorting.yaml.

    Attributes:
        num_bins: Number of physical sorting bins available.
        mechanism_type: Type of sorting mechanism (servo, stepper, conveyor, or TBD).
        bin_mapping: Mapping of part numbers to bin identifiers.
    """

    num_bins: int = 6
    mechanism_type: str = "TBD"
    bin_mapping: dict[str, int] = {}
