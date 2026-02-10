"""State machine definitions for the Lego Sorter application.

Defines the operational states, valid transitions, and configuration
for the sorting state machine per the Software Architecture Document.
"""

from dataclasses import dataclass
from enum import Enum, auto


class SorterState(Enum):
    """Operational states for the Lego Sorter.

    State flow:
        INITIALIZING -> IDLE -> RUNNING -> IDLE
        Any state -> ERROR -> IDLE | SHUTTING_DOWN
        IDLE -> SHUTTING_DOWN
    """

    INITIALIZING = auto()
    IDLE = auto()
    RUNNING = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


VALID_TRANSITIONS: dict[SorterState, list[SorterState]] = {
    SorterState.INITIALIZING: [SorterState.IDLE, SorterState.ERROR],
    SorterState.IDLE: [SorterState.RUNNING, SorterState.SHUTTING_DOWN],
    SorterState.RUNNING: [SorterState.IDLE, SorterState.ERROR],
    SorterState.ERROR: [SorterState.IDLE, SorterState.SHUTTING_DOWN],
    SorterState.SHUTTING_DOWN: [],
}


@dataclass
class SorterConfig:
    """Runtime configuration for the sorting cycle.

    Attributes:
        fade_duration: LED fade in/out time in seconds.
        action_duration: Motor action time in seconds.
        motor_delay: Delay between motor steps in seconds.
    """

    fade_duration: float = 0.5
    action_duration: float = 5.0
    motor_delay: float = 0.002
