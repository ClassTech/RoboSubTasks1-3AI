#!/usr/bin/env python3
"""
A simple task to command zero thrust, effectively shutting down motors.
"""
from typing import Tuple

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from data_structures import SensorSuite, Vision, ThrusterCommands
from config import SimulationConfig

class ShutdownTask(Task):
    """
    Immediately completes and commands zero thrust.
    """
    def __init__(self):
        # No target depth needed, motors off.
        super().__init__()
        self.subtasks = [] # No subtasks
        self.reset()

    def reset(self, search_direction: int = 1):
        super().reset(search_direction)
        # No specific reset needed

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                vision_data: Vision,
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        """Returns zero commands and completes."""
        print("INFO: ShutdownTask executing - commanding zero thrust.")
        # Return default ThrusterCommands (all zeros) and COMPLETED status
        return TaskStatus.COMPLETED, ThrusterCommands()