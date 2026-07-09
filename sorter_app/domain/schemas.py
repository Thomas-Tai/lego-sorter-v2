"""Pydantic schemas for LEGO Sorter V2 domain layer.

This module defines validated configuration models for the sorting mechanism.
All schemas use Pydantic v2 for runtime validation.
"""

from pydantic import BaseModel, Field


class BinInfo(BaseModel):
    """Information about a single sorting bin.

    Attributes:
        id: Unique bin identifier (0-indexed).
        x_mm: X coordinate of bin center in millimeters.
        y_mm: Y coordinate of bin center in millimeters.
        label: Human-readable label for the bin.
    """

    id: int = Field(ge=0, description="Unique bin identifier")
    x_mm: float = Field(ge=0.0, description="X coordinate in mm")
    y_mm: float = Field(ge=0.0, description="Y coordinate in mm")
    label: str = Field(min_length=1, description="Human-readable bin label")


class GridConfig(BaseModel):
    """Configuration for the bin grid layout.

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        bin_width_mm: Width of each bin in millimeters.
        bin_depth_mm: Depth of each bin in millimeters.
        bin_height_mm: Height of each bin in millimeters.
        x_spacing_mm: Spacing between bins in X direction.
        y_spacing_mm: Spacing between bins in Y direction.
        x_offset_mm: X offset from gantry home to grid origin.
        y_offset_mm: Y offset from gantry home to grid origin.
    """

    rows: int = Field(gt=0, description="Number of grid rows")
    cols: int = Field(gt=0, description="Number of grid columns")
    bin_width_mm: float = Field(gt=0.0, description="Bin width in mm")
    bin_depth_mm: float = Field(gt=0.0, description="Bin depth in mm")
    bin_height_mm: float = Field(gt=0.0, description="Bin height in mm")
    x_spacing_mm: float = Field(ge=0.0, description="X spacing between bins")
    y_spacing_mm: float = Field(ge=0.0, description="Y spacing between bins")
    x_offset_mm: float = Field(ge=0.0, description="Grid X offset from home")
    y_offset_mm: float = Field(ge=0.0, description="Grid Y offset from home")


class BinEntry(BaseModel):
    """Entry for a single bin in the grid.

    Attributes:
        id: Unique bin identifier.
        row: Row index in the grid.
        col: Column index in the grid.
        label: Human-readable label for the bin.
    """

    id: int = Field(ge=0, description="Unique bin identifier")
    row: int = Field(ge=0, description="Row index")
    col: int = Field(ge=0, description="Column index")
    label: str = Field(min_length=1, description="Bin label")


class OverflowConfig(BaseModel):
    """Configuration for the overflow/reject bin.

    The overflow bin is positioned outside the main grid and receives
    parts that cannot be classified or have low confidence.

    Attributes:
        id: Unique bin identifier (typically 8 for MVP).
        x_mm: X coordinate in millimeters.
        y_mm: Y coordinate in millimeters.
        label: Human-readable label.
    """

    id: int = Field(ge=0, description="Overflow bin ID")
    x_mm: float = Field(ge=0.0, description="X coordinate in mm")
    y_mm: float = Field(ge=0.0, description="Y coordinate in mm")
    label: str = Field(min_length=1, description="Overflow bin label")


class BinLayoutConfig(BaseModel):
    """Complete bin layout configuration.

    This is the root model for bin_layout.yaml, containing grid geometry,
    bin definitions, overflow bin, and part-to-bin assignments.

    Attributes:
        version: Configuration version for future compatibility.
        grid: Grid geometry configuration.
        bins: List of bin entries in the grid.
        overflow: Overflow bin configuration.
        assignments: Part-to-bin mapping (partId_colorId → bin_id).
        confidence_threshold: Minimum classification confidence required to
            route a part to its resolved bin; below this, callers should
            route to the overflow bin instead. Default (0.80) matches the
            legacy hardcoded CONFIDENCE_THRESHOLD constant previously in
            sorter_app/main.py and scripts/e2e_sort_simulation.py.
    """

    version: int = Field(ge=1, description="Config version")
    grid: GridConfig = Field(description="Grid geometry")
    bins: list[BinEntry] = Field(description="Grid bin entries")
    overflow: OverflowConfig = Field(description="Overflow bin config")
    assignments: dict[str, int] = Field(
        default_factory=dict,
        description="Part-to-bin mapping (partId_colorId → bin_id)",
    )
    confidence_threshold: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description=(
            "Minimum confidence to route to a resolved bin; below this, "
            "route to overflow"
        ),
    )


class GantrySerialConfig(BaseModel):
    """Serial port configuration for ESP32 communication.

    Attributes:
        port: Serial device path (e.g., /dev/ttyUSB0).
        baud_rate: UART baud rate (default: 115200).
        timeout_s: Per-command timeout in seconds.
        retry_count: Number of retries on timeout.
    """

    port: str = Field(min_length=1, description="Serial device path")
    baud_rate: int = Field(default=115200, gt=0, description="Baud rate")
    timeout_s: float = Field(default=5.0, gt=0.0, description="Timeout in seconds")
    retry_count: int = Field(default=2, ge=0, description="Retry count")


class GantryMotionConfig(BaseModel):
    """Motion parameters for the gantry.

    Attributes:
        x_max_mm: Maximum X travel in millimeters.
        y_max_mm: Maximum Y travel in millimeters.
        feed_rate_mm_min: Default feed rate in mm/min.
        homing_required: Whether homing is required before operation.
    """

    x_max_mm: float = Field(gt=0.0, description="Max X travel in mm")
    y_max_mm: float = Field(gt=0.0, description="Max Y travel in mm")
    feed_rate_mm_min: int = Field(default=3000, gt=0, description="Feed rate mm/min")
    homing_required: bool = Field(default=True, description="Homing required")


class GantryServoConfig(BaseModel):
    """Servo gate configuration.

    Attributes:
        open_angle_us: PWM pulse width for open position in microseconds.
        close_angle_us: PWM pulse width for closed position in microseconds.
        gate_hold_ms: Time to hold gate open in milliseconds.
    """

    open_angle_us: int = Field(default=2500, ge=500, le=2500, description="Open PWM us")
    close_angle_us: int = Field(
        default=500, ge=500, le=2500, description="Close PWM us"
    )
    gate_hold_ms: int = Field(default=300, ge=0, description="Hold time ms")


class PickupConfig(BaseModel):
    """Gantry pickup/handoff position configuration.

    Defines the gantry position used for part pickup/handoff with the
    conveyor, per spec.md §5.1/§5.2 (PickupConfig(x, y position)). Per
    SM-DES-004 §2.2b (Published Interface Ledger), the conveyor chute
    outlet is at X=0, which is also the gantry home position — so in the
    MVP mechanical design, pickup coincides with home (0.0, 0.0).

    NOTE: No consumer in the current codebase reads this config yet
    (GantrySortingService.sort_to_bin does not return-to-pickup between
    cycles; only GantrySortingService.cleanup() moves to a hardcoded
    (0.0, 0.0)). This section is schema/config groundwork for that future
    wiring — see UNVERIFIED notes in sw-2-report.md. Only x_mm/y_mm are
    defined because spec.md's PickupConfig has no other fields and no
    Z-axis or servo/settle-time concept exists anywhere else in the gantry
    design (the gantry is 2D X-Y only).

    Attributes:
        x_mm: X coordinate of the pickup position in millimeters.
        y_mm: Y coordinate of the pickup position in millimeters.
    """

    x_mm: float = Field(default=0.0, ge=0.0, description="Pickup X coordinate in mm")
    y_mm: float = Field(default=0.0, ge=0.0, description="Pickup Y coordinate in mm")


class GantrySimulationConfig(BaseModel):
    """Simulation mode configuration.

    Attributes:
        enabled: Whether simulation mode is enabled.
        move_delay_s: Simulated move delay in seconds.
    """

    enabled: bool = Field(default=False, description="Simulation mode enabled")
    move_delay_s: float = Field(default=0.1, ge=0.0, description="Simulated move delay")


class GantryConfig(BaseModel):
    """Complete gantry configuration.

    Root model for gantry.yaml, containing serial, motion, servo,
    and simulation settings.

    Attributes:
        serial: Serial port configuration.
        motion: Motion parameters.
        servo: Servo gate configuration.
        simulation: Simulation mode settings.
        pickup: Pickup/handoff position config. Optional with safe
            defaults (0.0, 0.0) so existing yaml-less GantryConfig(...)
            call sites and tests that predate this field keep working
            unchanged.
    """

    serial: GantrySerialConfig = Field(description="Serial config")
    motion: GantryMotionConfig = Field(description="Motion config")
    servo: GantryServoConfig = Field(description="Servo config")
    simulation: GantrySimulationConfig = Field(
        default_factory=GantrySimulationConfig, description="Simulation config"
    )
    pickup: PickupConfig = Field(
        default_factory=PickupConfig, description="Pickup/handoff position config"
    )
