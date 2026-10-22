#!/usr/bin/env python3
"""
Base class for reusable sub-actions within a larger Task.
"""
from enum import Enum, auto
from typing import Tuple, Dict, Any

# --- Ensure this line correctly imports Vision ---
from data_structures import SensorSuite, Vision, ThrusterCommands
# ---
from config import SimulationConfig
# Forward declaration no longer needed
# class Submarine: pass

class SubtaskStatus(Enum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

class Subtask:
    """Base class for a reusable action within a larger Task."""
    # --- Ensure type hints use Vision and 'Submarine' ---
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        pass

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        raise NotImplementedError

    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        pass
    # ---

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # Optional: For dynamic state names using context
    def get_dynamic_name(self, context: Dict[str, Any]) -> str:
        return self.name