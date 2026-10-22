#!/usr/bin/env python3
"""
Orbit Turn Task: Navigates around the marker using a dynamic orbit
(sway + yaw + surge controlled by width) until the gate is visible and pole is >= 80%.
"""
import math
from typing import Tuple, List, Dict, Any
import pygame
import numpy as np

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# Import all the building blocks
from ai.tasks.common_subtasks import (SwayStraight, WaitForTargetVisible, AlignToObjectX,
                                      ApproachAndCenterObject, Stabilize,
                                      DynamicOrbitPole, SwayUntilTargetLost)

from data_structures import SensorSuite, Vision, ThrusterCommands
from config import SimulationConfig
from controller import angle_diff

class OrbitTurnTask(Task):
    """
    Finds, approaches (centered), then orbits the pole (centered) using sway/yaw
    and surge (width PID) control until the gate is visible & pole is >= 80%.
    Then sways left until pole is lost.
    """
    def __init__(self,
                 target_depth: float = 1.0,
                 approach_height_fraction: float = 0.5, # Pole height as a fraction of image height
                 orbit_target_fraction: float = 0.5, # Keep pole centered during orbit
                 orbit_sway_power: float = -0.1,     # Slow sway right (positive = left)
                 orbit_yaw_gain: float = 2.5,
                 # --- NEW Width Parameters ---
                 orbit_target_width_fraction: float = 0.156, # Target width, fraction of image width (NEEDS TUNING)
                 orbit_width_p_gain: float = 16.0, # Gain for width-fraction error -> surge (NEEDS TUNING)
                 orbit_width_i_gain: float = 3.2,  # (NEEDS TUNING)
                 orbit_width_d_gain: float = 16.0, # (NEEDS TUNING)
                 orbit_width_i_clamp: float = 0.3, # (NEEDS TUNING)
                 # ---
                 final_sway_power: float = -0.3,     # Sway power for final move
                 final_extra_sway_seconds: float = 0.0):  # Keep swaying this long after the pole leaves the view

        self.target_depth = target_depth

        super().__init__() # Call base constructor

        self.subtasks = [
            # 1. Find the pole
            WaitForTargetVisible(target_type='pole'),

            # 2. Align to center (50%)
            AlignToObjectX(
                target_x_fraction=0.5,
                tolerance_fraction=0.047,
                yaw_gain=orbit_yaw_gain,
                yaw_rate_tolerance=0.05
            ),

            # 3. Get close to it while keeping it CENTERED
            ApproachAndCenterObject(
                height_threshold_fraction=approach_height_fraction, # Use height for initial approach
                target_x_fraction=0.5, # Keep centered
                surge_p_gain=24.0,
                height_tolerance_fraction=0.0625,
                yaw_gain=orbit_yaw_gain,
                align_tolerance_fraction=0.0625
            ),

            # 4. Stop briefly
            Stabilize(duration=1.0),

            # 5. Execute the dynamic orbit keeping pole CENTERED until gate visible & pole >= 80%
            DynamicOrbitPole(
                target_x_fraction=orbit_target_fraction, # Use 0.5 here
                sway_power=orbit_sway_power,
                yaw_gain=orbit_yaw_gain,
                # --- Pass WIDTH parameters ---
                target_pole_width_fraction=orbit_target_width_fraction,
                orbit_width_p_gain=orbit_width_p_gain,
                orbit_width_i_gain=orbit_width_i_gain,
                orbit_width_d_gain=orbit_width_d_gain,
                orbit_width_i_clamp=orbit_width_i_clamp
                # ---
            ),

            # 6. Stabilize after orbit completes
            Stabilize(duration=1.0),

            # 7. Sway left until the POLE is lost
            SwayUntilTargetLost(
                sway_power=final_sway_power,
                target_type='pole'
            ),

            # 8. Stabilize after pole is lost (ready for GateTask)
            Stabilize(duration=1.0)
        ]
        if final_extra_sway_seconds > 0:
            # Extra sideways distance, so the drive back to the gate clears the pole
            self.subtasks.insert(-1, SwayStraight(duration=final_extra_sway_seconds, sway_power=final_sway_power))
        self.reset()

    def reset(self, search_direction: int = 1):
        """Resets the task and adds target_depth to the context."""
        super().reset(search_direction)
        self.context['target_depth'] = self.target_depth

    # No special execute() logic needed.