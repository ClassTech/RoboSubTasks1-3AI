#!/usr/bin/env python3
"""
A simple task to bring the submarine to a full stop using PID control.
Now holds 3D position (X, Y, Z) and Heading.
"""
import math
from typing import Tuple

from .task_base import Task, TaskStatus
# --- CORRECT IMPORT ---
from data_structures import SensorSuite, Vision, ThrusterCommands
# ---
from config import SimulationConfig

class StabilizeTask(Task):
    """Holds 3D position (X,Y,Z) and heading using PID control."""
    def __init__(self, duration: float = 3.0, 
                 speed_threshold: float = 0.05,
                 target_depth: float = 1.0): # <-- ADDED
        
        self.target_depth = target_depth # <-- ADDED
        self.STABILIZE_DURATION = duration
        self.SPEED_THRESHOLD = speed_threshold
        
        # Internal state
        self.state_timer = 0.0
        self.target_set = False
        self._current_speed = 0.0 
        self._current_speed_z = 0.0
        super().__init__() 
        self.reset() 

    def reset(self, search_direction: int = 1): # Match base signature
        super().reset(search_direction) 
        self.state_timer = 0.0
        self.target_set = False
        self._current_speed = 0.0
        self._current_speed_z = 0.0
        # Add depth to context (though this task doesn't use subtasks)
        self.context['target_depth'] = self.target_depth

    @property
    def state_name(self) -> str:
        speed_xy = getattr(self, '_current_speed', 0.0)
        speed_z = getattr(self, '_current_speed_z', 0.0)
        return f"STABILIZING (XY: {speed_xy:.2f} Z: {speed_z:.2f})"

    # process_vision is removed

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, 
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        
        # On first execution, lock 3D position/heading
        if not self.target_set:
            sub.target_x, sub.target_y = sensors.x, sensors.y
            # Use the depth passed during construction
            sub.target_depth = self.target_depth 
            sub.target_heading = sensors.heading
            sub.target_pitch = 0.0 
            
            sub.integral_x_err, sub.integral_y_err = 0.0, 0.0
            self.target_set = True

        self.state_timer += dt
        speed_xy = math.hypot(sensors.velocity_x, sensors.velocity_y)
        speed_z = abs(sensors.depth_rate)
        self._current_speed = speed_xy
        self._current_speed_z = speed_z

        # Check completion
        if (self.state_timer > self.STABILIZE_DURATION and 
            speed_xy < self.SPEED_THRESHOLD and 
            speed_z < self.SPEED_THRESHOLD):
            
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)

        # Execute 3D PID hover control
        return TaskStatus.RUNNING, sub._get_pid_hover_commands(sensors, dt, 
                                                               sub.target_x, sub.target_y, 
                                                               self.target_depth)