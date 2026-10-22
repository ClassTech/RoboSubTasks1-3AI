#!/usr/bin/env python3
"""
Base class definition for all autonomous mission tasks.
Includes subtask execution logic with a shared context.
Handles subtask failure by restarting the task sequence.
Allows tasks to specify search direction on restart.
"""
from enum import Enum, auto
from typing import Tuple, List, Dict, Any

# --- Import Vision class instead of VisionData ---
from data_structures import SensorSuite, Vision, ThrusterCommands
# ---
from config import SimulationConfig
from ai.tasks.subtask_base import Subtask, SubtaskStatus # Keep this import
# --- NO IMPORT from common_subtasks here ---

class TaskStatus(Enum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto() # Propagated up if task cannot recover

class Task:
    """Base class for a mission task with subtask execution."""
    subtasks: List[Subtask] = []
    current_subtask_index: int = 0
    DEFAULT_SEARCH_TURN_POWER = 0.25
    context: Dict[str, Any] = {}
    search_direction: int = 1
    target_depth: float = 0.1

    def reset(self, search_direction: int = 1):
        self.current_subtask_index = 0
        self.context = {}
        self.search_direction = search_direction
        if not 'target_depth' in self.context:
            self.context['target_depth'] = getattr(self, 'target_depth', 0.1)
        for subtask in self.subtasks:
             if hasattr(subtask, '_has_entered'): delattr(subtask, '_has_entered')
             if hasattr(subtask, 'reset'): subtask.reset()

    @property
    def state_name(self) -> str:
        if self.subtasks and 0 <= self.current_subtask_index < len(self.subtasks):
            subtask_name = self.subtasks[self.current_subtask_index].name
            if hasattr(self.subtasks[self.current_subtask_index], 'get_dynamic_name'):
                subtask_name = self.subtasks[self.current_subtask_index].get_dynamic_name(self.context)
            return f"{self.__class__.__name__}[{self.current_subtask_index}]:{subtask_name}"
        return self.__class__.__name__

    # --- USE STRING HINT 'Submarine' ---
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                vision_data: Vision,
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        if not self.subtasks: return TaskStatus.COMPLETED, ThrusterCommands()
        if self.current_subtask_index >= len(self.subtasks): return TaskStatus.COMPLETED, ThrusterCommands()

        processed_vision_data = vision_data
        current_subtask = self.subtasks[self.current_subtask_index]
        self.context['search_direction'] = self.search_direction
        self.context['search_turn_power'] = self.DEFAULT_SEARCH_TURN_POWER

        if not hasattr(current_subtask, '_has_entered'):
             current_subtask.on_enter(sub, sensors, processed_vision_data, self.context)
             current_subtask._has_entered = True

        subtask_status, commands = current_subtask.execute(sub, dt, sensors, processed_vision_data, config, self.context)

        if subtask_status == SubtaskStatus.COMPLETED:
            current_subtask.on_exit(sub, sensors, processed_vision_data, self.context)
            delattr(current_subtask, '_has_entered')
            self.current_subtask_index += 1
            if self.current_subtask_index >= len(self.subtasks):
                is_empty_command = (commands == ThrusterCommands())
                final_commands = sub._get_damping_commands(sensors) if is_empty_command else commands
                return TaskStatus.COMPLETED, final_commands
            else:
                 next_subtask = self.subtasks[self.current_subtask_index]
                 next_subtask.on_enter(sub, sensors, processed_vision_data, self.context)
                 next_subtask._has_entered = True
                 return TaskStatus.RUNNING, ThrusterCommands()
        elif subtask_status == SubtaskStatus.FAILED:
            print(f"ERROR: Subtask {current_subtask.name} FAILED in task {self.__class__.__name__}")
            current_subtask.on_exit(sub, sensors, processed_vision_data, self.context)
            if hasattr(current_subtask, '_has_entered'):
                delattr(current_subtask, '_has_entered')
            return TaskStatus.FAILED, sub._get_damping_commands(sensors)
        return TaskStatus.RUNNING, commands