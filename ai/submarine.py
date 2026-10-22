#!/usr/bin/env python3
"""
Contains the "brain" of the sub: the Submarine class that manages the mission plan and AI logic.
"""
import math
import numpy as np
from typing import Tuple, List, Optional

# --- Import Vision class ---
from data_structures import ThrusterCommands, SensorSuite, Vision
# ---
from config import SimulationConfig
from controller import ControllerGains, angle_diff, depth_effort, yaw_effort, mix
from ai.tasks import TaskStatus # Corrected import
from ai.tasks.task_base import Task

class Submarine:
    def __init__(self, mission_plan: List[Task]):
        # === PID and Control Gains ===
        
        # --- Horizontal (XY) Gains ---
        self.ALIGN_YAW_P_GAIN = 0.6
        self.YAW_D_GAIN = 4.0 
        self.HOVER_YAW_P_GAIN = 0.1
        self.HOVER_XY_P_GAIN = 1.0
        self.HOVER_XY_I_GAIN = 0.0 
        self.HOVER_XY_D_GAIN = 4.0
        self.DAMPING_GAIN = 1.5
        self.MANEUVER_DAMPING_GAIN = 3.0 

       # --- Vertical (Z/Pitch) Gains (TUNE ME) ---
        # --- Adjusted P and I gains ---
        # Retuned for the measured thrust and drag; see controller.ControllerGains
        self.DEPTH_P_GAIN = 9.0
        self.DEPTH_I_GAIN = 0.1
        self.DEPTH_D_GAIN = 3.0
        
        # Integrals
        self.integral_x_err = 0.0
        self.integral_y_err = 0.0
        self.integral_z_err = 0.0 
        self.integral_clamp = 2.0 
        self.integral_clamp_z = 2.0 
        # ---

        self.MIN_PIXELS_FOR_DETECTION = 20

        self.mission_plan = mission_plan
        self.config = SimulationConfig()
        
        self._latest_sensors: Optional[SensorSuite] = None
        
        self.vision = Vision(image_provider=lambda: self._latest_sensors.camera_image if self._latest_sensors else None,
                             min_pole_pixels=self.MIN_PIXELS_FOR_DETECTION,
                             min_gate_pixels=50,
                             detection_provider=lambda: self._latest_sensors.detections if self._latest_sensors else None,
                             image_width=self.config.cameraWidth,
                             image_height=self.config.cameraHeight)
        
        self.reset()

    def reset(self):
        self.current_task_index = 0
        for task in self.mission_plan:
            task.reset()
        # --- 3D Targets ---
        self.target_x, self.target_y = 0.0, 0.0
        self.target_heading = 0.0
        self.target_depth = 0.0 
        self.target_pitch = 0.0
        # ---
        self.integral_x_err, self.integral_y_err = 0.0, 0.0 
        self.integral_z_err = 0.0 
        self._latest_sensors = None 
        if hasattr(self, 'vision'): self.vision.reset()

    def update(self, dt: float, sensors: SensorSuite) -> Tuple[ThrusterCommands, Vision]:

        self._latest_sensors = sensors

        # --- MODIFIED: Check for mission completion FIRST ---
        if self.current_task_index >= len(self.mission_plan):
            # Mission is fully complete. Return ZERO commands.
            return ThrusterCommands(), self.vision
        # ---

        current_task = self.mission_plan[self.current_task_index]
        self.target_depth = getattr(current_task, 'target_depth', self.target_depth)
        self.vision.update(dt)

        depth_error = self.target_depth - sensors.depth
        self.integral_z_err = np.clip(self.integral_z_err + depth_error * dt, -self.integral_clamp_z, self.integral_clamp_z)

        status, commands = current_task.execute(self, dt, sensors, self.vision, self.config)

        # --- Task Transition Logic ---
        # --- MODIFIED: Allow transition even for the last task ---
        if status == TaskStatus.COMPLETED:
            # Check if it was the *very last* task completing
            if self.current_task_index >= len(self.mission_plan) - 1:
                print("INFO: Final mission task completed.")
                self.current_task_index += 1 # Increment index past the end
                # Return the final commands from the completed task (e.g., zero from Shutdown)
                return commands, self.vision
            else:
                # Standard transition to the next task
                self.current_task_index += 1
                next_task = self.mission_plan[self.current_task_index]
                self.target_depth = getattr(next_task, 'target_depth', self.target_depth)
                if hasattr(next_task, 'SPEED_THRESHOLD') or next_task.__class__.__name__ == 'DiveToDepth':
                    print(f"INFO: Resetting PID integrals for task: {next_task.__class__.__name__}")
                    self.integral_x_err, self.integral_y_err, self.integral_z_err = 0.0, 0.0, 0.0
                # Call on_enter for the next task if it exists (using correct method name)
                if hasattr(next_task, 'on_enter'):
                     # Need vision data for on_enter - re-get? No, use current frame's
                     next_task.on_enter(self, sensors, self.vision, next_task.context)
                     # Set _has_entered manually since base execute won't run yet
                     if hasattr(next_task, 'subtasks') and next_task.subtasks:
                         if next_task.current_subtask_index < len(next_task.subtasks):
                              next_task.subtasks[next_task.current_subtask_index]._has_entered = True


                # Return zero commands during the transition frame
                return ThrusterCommands(), self.vision
        # --- END MODIFIED TRANSITION ---
        elif status == TaskStatus.FAILED:
            print(f"ERROR: Task {current_task.__class__.__name__} FAILED. Stopping.")
            # Optionally, implement a failure state or stop simulation
            # For now, just damp and effectively end mission by incrementing index
            self.current_task_index = len(self.mission_plan) # Move index past end
            return self._get_damping_commands(sensors), self.vision

        # Task still running
        return commands, self.vision
    
    # --- ADDED METHODS BACK ---
    def get_current_task_name(self) -> str:
        """Returns the class name of the currently running task."""
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].__class__.__name__
        return "MISSION_COMPLETE"

    def get_current_state_name(self) -> str:
        """Returns the specific state name within the current task."""
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].state_name
        return ""
    # --- END ADDED METHODS ---

    # ---
    # === NEW/UPDATED CONTROL HELPERS ===
    # ---

    def get_heave_command(self, sensors: SensorSuite, target_depth: float) -> float:
        """Heave (vertical) command from the depth PID. Pitch isn't controlled: the vertical
        thrusters are on the sides, and the sub is passively stable in pitch."""
        return depth_effort(self.gains(), sensors.depth, sensors.depth_rate, self.integral_z_err, target_depth)

    def gains(self) -> ControllerGains:
        """This sub's gains in the shared controller's form."""
        return ControllerGains(depth_p=self.DEPTH_P_GAIN, depth_i=self.DEPTH_I_GAIN, depth_d=self.DEPTH_D_GAIN,
                               depth_i_clamp=self.integral_clamp_z, heading_p=self.HOVER_YAW_P_GAIN,
                               yaw_d=self.YAW_D_GAIN)

    def _get_damping_commands(self, sensors: SensorSuite) -> ThrusterCommands:
        """Fully damps all 6 axes of motion.
        SIM ONLY: uses ground-truth XY velocity, so it can't run on the real sub."""
        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad = math.radians(sensors.heading)
        c, s = math.cos(h_rad), math.sin(h_rad)
        surge, sway = wdx*c+wdy*s, -wdx*s+wdy*c
        
        yaw = -sensors.imu.gyro_z * self.YAW_D_GAIN 
        
        # Hold current depth
        heave = self.get_heave_command(sensors, sensors.depth)
        
        return self._mix_and_normalize_commands(surge, sway, yaw, heave)

    def get_spin_damping_commands(self, sensors: SensorSuite, target_depth: Optional[float] = None):
        """Damps everything *except* yaw, while holding depth.
        SIM ONLY: uses ground-truth XY velocity, so it can't run on the real sub."""
        target_depth = target_depth if target_depth is not None else self.target_depth

        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad, cos_h, sin_h = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        surge, sway = wdx * cos_h + wdy * sin_h, -wdx * sin_h + wdy * cos_h
        
        heave = self.get_heave_command(sensors, target_depth)
        
        return self._mix_and_normalize_commands(surge, sway, 0.0, heave)

    def get_heading_commands(self, sensors, heading, 
                             surge_power=0.0, 
                             sway_power=0.0, 
                             target_depth: Optional[float] = None,
                             target_pitch: float = 0.0):
        """Holds heading and depth/pitch (now with PID), applies surge/sway power.
        Uses only measured values, so the policy can drive the real sub through it."""
        target_depth = target_depth if target_depth is not None else self.target_depth

        yaw = yaw_effort(self.gains(), sensors.heading, sensors.imu.gyro_z, heading)
        
        # Open-loop sway: the real sub can't measure sway velocity to damp it.
        sway = np.clip(sway_power, -1.0, 1.0)
        
        # Vertical Control
        heave = self.get_heave_command(sensors, target_depth)
        
        return self._mix_and_normalize_commands(surge_power, sway, yaw, heave)

    def _get_pid_hover_commands(self, sensors: SensorSuite, dt: float, 
                                tx: float, ty: float, 
                                tz: float, 
                                yaw_p_gain_override: Optional[float] = None) -> ThrusterCommands:
        """Holds an XY position. SIM ONLY: uses ground-truth position and velocity."""
        # --- Horizontal XY Control (PID) ---
        yaw_p_gain = yaw_p_gain_override if yaw_p_gain_override is not None else self.HOVER_YAW_P_GAIN
        ex, ey = tx - sensors.x, ty - sensors.y
        self.integral_x_err = np.clip(self.integral_x_err + ex * dt, -self.integral_clamp, self.integral_clamp)
        self.integral_y_err = np.clip(self.integral_y_err + ey * dt, -self.integral_clamp, self.integral_clamp)
        wtx = (ex * self.HOVER_XY_P_GAIN) + (self.integral_x_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_x * self.HOVER_XY_D_GAIN)
        wty = (ey * self.HOVER_XY_P_GAIN) + (self.integral_y_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_y * self.HOVER_XY_D_GAIN)
        h_rad, c, s = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        surge, sway = wtx * c + wty * s, -wtx * s + wty * c
        
        # --- Yaw Control (PD) ---
        yaw_err = angle_diff(self.target_heading, sensors.heading)
        yaw_p = yaw_err * yaw_p_gain
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw_cmd = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        # --- Vertical Z/Pitch Control (PID for Z, PD for Pitch) ---
        heave = self.get_heave_command(sensors, tz)
        
        return self._mix_and_normalize_commands(surge, sway, yaw_cmd, heave)

    def _mix_and_normalize_commands(self, surge, sway, yaw, heave) -> ThrusterCommands:
        """Mixes 4-axis commands into 6 thruster commands (shared controller.mix)."""
        return mix(surge, sway, yaw, heave)
