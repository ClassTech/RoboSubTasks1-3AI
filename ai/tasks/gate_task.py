#!/usr/bin/env python3
"""
Refactored Gate task using reusable subtasks. Includes stabilization after alignment.
Vision processing is now handled by the Vision class passed from Submarine.
Now passes target_depth to context.
"""
import math
from typing import Tuple, List
import pygame
import numpy as np

# --- USE STRING HINT 'Submarine' ---
from ai.tasks.task_base import Task, TaskStatus # Keep Task import
# ---
from ai.tasks.subtask_base import Subtask, SubtaskStatus
from ai.tasks.common_subtasks import (DiveToDepth, WaitForTargetVisible, AlignToObjectX, DriveStraight,
                                      Stabilize, DriveUntilTargetLost)

from data_structures import SensorSuite, Vision, ThrusterCommands
from controller import angle_diff

class GateTask(Task):
    """Navigates through the gate defined by two red poles at a target depth."""

    def __init__(self, target_depth: float = 1.0):

        self.target_depth = target_depth # Store target depth

        self.subtasks = [
            DiveToDepth(),
            WaitForTargetVisible(target_type='gate'),
            AlignToObjectX(target_x_fraction=0.5,
                           tolerance_fraction=0.031,
                           yaw_gain=1.5,
                           yaw_rate_tolerance=0.05),
            Stabilize(duration=1.0, speed_threshold=0.1),
            DriveUntilTargetLost(surge_power=0.6, target_type='gate'),
            DriveStraight(duration=12.0, surge_power=0.6),   # ~3 m at the measured thrust and drag (was 4 s)
            Stabilize(duration=1.0)
        ]

        super().__init__()
        self.reset()

    def reset(self, search_direction: int = 1):
        """Resets the task and adds target_depth to the context."""
        super().reset(search_direction)
        self.context['target_depth'] = self.target_depth

    # Execute method is inherited, uses 'Submarine' hint from base class