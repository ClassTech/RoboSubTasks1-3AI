#!/usr/bin/env python3
"""
Implementations of common, reusable Subtasks, using context.
Subtasks now expect a Vision object and hold depth.
"""
import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Absolute import for the base class
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# --- Import Vision class ---
from data_structures import SensorSuite, Vision, ThrusterCommands
# ---
from config import SimulationConfig
from controller import angle_diff
from perception import CameraModel, bearing_deg
# --- NO IMPORT from task_base here ---

# --- DiveToDepth Subtask ---
# ... (DiveToDepth code remains the same) ...
class DiveToDepth(Subtask):
    def __init__(self, depth_tolerance: float = 0.15, z_vel_tolerance: float = 0.05):
        self.depth_tolerance = depth_tolerance
        self.z_vel_tolerance = z_vel_tolerance
        self.target_depth: float = 1.0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.target_heading: float = 0.0
        self.target_set = False
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.target_depth = context.get('target_depth', sensors.depth)
        self.target_x = sensors.x
        self.target_y = sensors.y
        self.target_heading = sensors.heading
        sub.target_x = self.target_x; sub.target_y = self.target_y; sub.target_heading = self.target_heading
        sub.integral_x_err = 0.0; sub.integral_y_err = 0.0; sub.integral_z_err = 0.0
        self.target_set = True
        print(f"INFO: DiveToDepth starting. Target: {self.target_depth:.2f}m")
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if not self.target_set: self.on_enter(sub, sensors, vision_data, context)
        depth_error = self.target_depth - sensors.depth
        if (abs(depth_error) < self.depth_tolerance and abs(sensors.depth_rate) < self.z_vel_tolerance):
            print("INFO: DiveToDepth complete."); return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub._get_pid_hover_commands(sensors, dt, self.target_x, self.target_y, self.target_depth)
        return SubtaskStatus.RUNNING, commands
    def get_dynamic_name(self, context: Dict[str, Any]) -> str: return f"{self.name}({self.target_depth:.1f}m)"

# --- SwayStraight Subtask ---
# ... (SwayStraight code remains the same) ...
class SwayStraight(Subtask):
    def __init__(self, duration: float, sway_power: float = 0.5):
        self.duration = duration; self.sway_power = sway_power; self.timer = 0.0
        self.heading_to_hold = None; self.target_depth: float = 0.1
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.timer = self.duration; self.heading_to_hold = context.get('initial_heading', sensors.heading)
        self.target_depth = context.get('target_depth', sensors.depth)
        print(f"INFO: SwayStraight starting. Power: {self.sway_power}, Duration: {self.duration}")
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.timer <= 0: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        self.timer -= dt
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        commands = sub.get_heading_commands(sensors, self.heading_to_hold, surge_power=0.0, sway_power=self.sway_power, target_depth=self.target_depth)
        return SubtaskStatus.RUNNING, commands
    def get_dynamic_name(self, context: Dict[str, Any]) -> str: return f"{self.name}({self.sway_power:+.1f})"

# --- MODIFIED DynamicOrbitPole Subtask ---
class DynamicOrbitPole(Subtask):
    """
    Orbits the pole by applying constant sway and using yaw control
    to keep the pole at a target X fraction. Uses PID surge control based
    on apparent WIDTH (as a fraction of image width) to maintain distance.
    Completes when the gate is visible AND the pole is >= 80% screen width.
    """
    def __init__(self,
                 target_x_fraction: float = 0.5,
                 sway_power: float = -0.3,
                 yaw_gain: float = 1.5,
                 # --- CHANGED to WIDTH ---
                 target_pole_width_fraction: float = 0.125, # Target apparent width, fraction of image width
                 orbit_width_p_gain: float = 16.0, # P gain for width error -> surge (NEEDS TUNING)
                 orbit_width_i_gain: float = 3.2,  # I gain (NEEDS TUNING)
                 orbit_width_d_gain: float = 16.0, # D gain (NEEDS TUNING)
                 orbit_width_i_clamp: float = 0.3, # Max integral contribution
                 # ---
                 lost_timeout: float = 3.0,
                 local_search_yaw: float = -0.2):

        self.target_x_fraction = target_x_fraction
        self.sway_power = sway_power
        self.yaw_gain = yaw_gain
        # Width PID parameters
        self.target_pole_width_fraction = target_pole_width_fraction
        self.width_p_gain = orbit_width_p_gain
        self.width_i_gain = orbit_width_i_gain
        self.width_d_gain = orbit_width_d_gain
        self.width_i_clamp = orbit_width_i_clamp
        # ---
        self.lost_timeout = lost_timeout
        self.local_search_yaw = local_search_yaw
        self.target_depth: float = 0.1
        self.time_since_target_lost = 0.0
        self.integral_width_err = 0.0 # Width integral term
        self.last_width_error = 0.0   # For D term calculation

    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.target_depth = context.get('target_depth', sensors.depth)
        self.time_since_target_lost = 0.0
        self.integral_width_err = 0.0 # Reset integral
        self.last_width_error = 0.0
        # Get initial width error
        pole_blob = vision_data.get_best_pole()
        if pole_blob:
            cam_w, _ = sensors.camera_image.get_size()
            self.last_width_error = self.target_pole_width_fraction - pole_blob['width'] / cam_w
        else:
             self.last_width_error = 0.0 # Assume no error if not visible yet

        print(f"INFO: DynamicOrbitPole starting. Target X: {self.target_x_fraction*100:.0f}%, Sway: {self.sway_power}, Target Width: {self.target_pole_width_fraction:.3f} of image")

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        pole_visible = vision_data.is_pole_visible()
        pole_blob = vision_data.get_best_pole() # Get the full blob data
        pole_center_x = pole_blob['center_x'] if pole_blob else None
        gate_visible = vision_data.is_gate_visible()

        cam_w, _ = sensors.camera_image.get_size()
        current_width = pole_blob['width'] / cam_w if pole_blob else 0.0
        target_pole_x_px = cam_w * self.target_x_fraction

        # --- Completion Check: Gate Visible AND Gate Right of Pole? ---
        gate_right_of_pole = False
        gate_center_x = vision_data.get_gate_center_x()
        if gate_visible and pole_visible and gate_center_x is not None and pole_center_x is not None:
            gate_right_of_pole = (gate_center_x > pole_center_x)

        # Require gate visibility AND it being right of the pole
        if gate_visible and pole_visible and gate_right_of_pole:
            print(f"INFO: DynamicOrbitPole found gate ({gate_visible}) right of pole ({gate_right_of_pole}). Completing.")
            context['initial_heading'] = sensors.heading # Store heading
            return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)

        if pole_visible and pole_center_x is not None:
            # --- Pole is visible: Execute Orbit Control ---
            self.time_since_target_lost = 0.0

            # Yaw P-D Controller (to keep pole at target_x_fraction)
            pixel_error_x = pole_center_x - target_pole_x_px
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw_cmd = np.clip(yaw_p + yaw_d, -1.0, 1.0)

            # Constant Sway Power
            sway_cmd = self.sway_power

            # --- NEW: Surge PID Controller based on WIDTH ---
            width_error = self.target_pole_width_fraction - current_width
            # Update integral term
            self.integral_width_err += width_error * dt
            self.integral_width_err = np.clip(self.integral_width_err, -self.width_i_clamp, self.width_i_clamp)
            # Calculate derivative of error
            error_rate = (width_error - self.last_width_error) / dt if dt > 0 else 0.0
            self.last_width_error = width_error # Store for next iteration
            # Calculate PID terms
            surge_p = width_error * self.width_p_gain
            surge_i = self.integral_width_err * self.width_i_gain
            surge_d = error_rate * self.width_d_gain # D acts on rate of change of error
            # Combine terms (Positive error = too far = positive surge)
            surge_cmd = surge_p + surge_i + surge_d
            surge_cmd = np.clip(surge_cmd, -0.5, 0.5)
            # ---

            # Hold Depth
            heave_cmd = sub.get_heave_command(sensors, self.target_depth)

            # Mix commands
            commands = sub._mix_and_normalize_commands(surge_cmd, sway_cmd, yaw_cmd, heave_cmd)
            return SubtaskStatus.RUNNING, commands
        else:
            # --- Pole is LOST: Local Search or Fail ---
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout:
                print(f"ERROR: DynamicOrbitPole failed - pole lost for > {self.lost_timeout}s")
                return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
            # Spin to search locally, stop surge/sway
            yaw_cmd = self.local_search_yaw
            sway_cmd = 0.0
            surge_cmd = 0.0
            self.integral_width_err = 0.0 # Reset integral
            self.last_width_error = 0.0
            # Hold Depth
            heave_cmd = sub.get_heave_command(sensors, self.target_depth)
            commands = sub._mix_and_normalize_commands(surge_cmd, sway_cmd, yaw_cmd, heave_cmd)
            return SubtaskStatus.RUNNING, commands

    def get_dynamic_name(self, context: Dict[str, Any]) -> str:
        search_indicator = f" Searching! ({self.time_since_target_lost:.1f}s)" if self.time_since_target_lost > 0 else ""
        return f"{self.name}({self.target_x_fraction*100:.0f}%){search_indicator}"

# --- SwayUntilTargetLost Subtask ---
# ... (SwayUntilTargetLost code remains the same) ...
class SwayUntilTargetLost(Subtask):
    """
    Sways straight (holding heading and depth) until the specified
    vision target (default='pole') leaves the view. It only counts as gone if
    it was last seen near an image edge; a miss in mid-image (a dropout) just
    keeps the sway going, failing after lost_timeout seconds.
    """
    def __init__(self, sway_power: float = -0.3, target_type: str = 'pole',
                 edge_margin_fraction: float = 0.1, lost_timeout: float = 5.0):
        if target_type not in ['gate', 'pole', 'either']:
             raise ValueError("target_type must be 'gate', 'pole', or 'either'")
        self.sway_power = sway_power # Negative = right
        self.target_type = target_type
        self.edge_margin_fraction = edge_margin_fraction
        self.lost_timeout = lost_timeout
        self.heading_to_hold: Optional[float] = None
        self.target_was_visible = False
        self.last_near_edge = False
        self.time_unseen = 0.0
        self.target_depth: float = 0.1

    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 vision: Vision,
                 context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading)
        self.target_depth = context.get('target_depth', sensors.depth)
        self.last_near_edge = False
        self.time_unseen = 0.0
        if self.target_type == 'gate':
             self.target_was_visible = vision.is_gate_visible()
        elif self.target_type == 'pole':
             self.target_was_visible = vision.is_pole_visible()
        else: # either
             self.target_was_visible = vision.is_gate_visible() or vision.is_pole_visible()
        if not self.target_was_visible:
            print(f"WARN: SwayUntilTargetLost starting but target '{self.target_type}' not visible!")
        else:
            print(f"INFO: SwayUntilTargetLost starting. Target '{self.target_type}' visible. Sway Power: {self.sway_power}")

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None:
             print("WARN: SwayUntilTargetLost heading_to_hold is None, using current.")
             self.heading_to_hold = sensors.heading
        # Horizontal extent of the target in the image, if visible
        extent = None
        if self.target_type in ('gate', 'either') and vision.is_gate_visible():
             left, right = vision.get_gate_pair()
             extent = (left['min_x'], right['max_x'])
        elif self.target_type in ('pole', 'either') and vision.is_pole_visible():
             pole = vision.get_best_pole()
             extent = (pole['min_x'], pole['max_x'])
        commands = sub.get_heading_commands(sensors, self.heading_to_hold,
                                            surge_power=0.0,
                                            sway_power=self.sway_power,
                                            target_depth=self.target_depth)
        if extent is not None:
            cam_w, _ = sensors.camera_image.get_size()
            margin = self.edge_margin_fraction * cam_w
            self.target_was_visible = True
            self.last_near_edge = extent[0] < margin or extent[1] > cam_w - margin
            self.time_unseen = 0.0
            return SubtaskStatus.RUNNING, commands
        self.time_unseen += dt
        if self.target_was_visible and self.last_near_edge:
            print(f"INFO: SwayUntilTargetLost complete. Target '{self.target_type}' left the view.")
            context['initial_heading'] = sensors.heading
            return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        if self.time_unseen > self.lost_timeout:
            print(f"ERROR: SwayUntilTargetLost failed. Target '{self.target_type}' not seen for {self.lost_timeout}s.")
            return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
        return SubtaskStatus.RUNNING, commands

    def get_dynamic_name(self, context: Dict[str, Any]) -> str:
        return f"{self.name}({self.sway_power:+.1f})"


# --- Navigation Subtasks ---
class DriveStraight(Subtask):
    def __init__(self, duration: float, surge_power: float = 0.5):
        self.duration=duration; self.surge_power=surge_power; self.timer=0.0
        self.heading_to_hold=None; self.target_depth: float = 0.1
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.timer = self.duration; self.heading_to_hold = context.get('initial_heading', sensors.heading)
        self.target_depth = context.get('target_depth', sensors.depth)
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.timer <= 0: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        self.timer -= dt
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power, target_depth=self.target_depth)
        return SubtaskStatus.RUNNING, commands

class Stabilize(Subtask):
    def __init__(self, duration: float = 2.0, speed_threshold: float = 0.05):
        self.duration=duration; self.speed_threshold=speed_threshold; self.timer=0.0
        self.target_set=False; self.target_depth: float = 0.1
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.timer=0.0; self.target_set=False; self.target_depth = context.get('target_depth', sensors.depth)
        sub.target_x = sensors.x; sub.target_y = sensors.y; sub.target_heading = sensors.heading; sub.target_depth = self.target_depth
        sub.integral_x_err = 0.0; sub.integral_y_err = 0.0; sub.integral_z_err = 0.0
        self.target_set = True
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if not self.target_set: self.on_enter(sub, sensors, vision_data, context)
        self.timer += dt
        speed_xy = math.hypot(sensors.velocity_x, sensors.velocity_y); speed_z = abs(sensors.depth_rate)
        if self.timer > self.duration and speed_xy < self.speed_threshold and speed_z < self.speed_threshold: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y, sub.target_depth)
        return SubtaskStatus.RUNNING, commands

# --- Vision-Based Subtasks ---

class DriveUntilTargetLost(Subtask):
    """
    Drives forward, steering at the target's center while it's in view and
    holding the last heading when it isn't, until we're passing the target:
    - it disappears after reaching the image edges (we're right at it), or
    - it disappears after looking big (close) and stays gone for coast_seconds
      (we went through it during missed detections).
    If it disappears while still small (far away), keeps driving and only fails
    after lost_timeout seconds without seeing it.
    """
    def __init__(self, surge_power: float = 0.5, target_type: str = 'gate',
                 edge_margin_fraction: float = 0.1, pass_size_fraction: float = 0.3,
                 coast_seconds: float = 3.0, lost_timeout: float = 12.0):
        if target_type not in ['gate', 'pole', 'either']: raise ValueError("target_type must be 'gate', 'pole', or 'either'")
        self.surge_power = surge_power; self.target_type = target_type
        self.edge_margin_fraction = edge_margin_fraction; self.pass_size_fraction = pass_size_fraction
        self.coast_seconds = coast_seconds; self.lost_timeout = lost_timeout
        self.heading_to_hold: Optional[float] = None; self.target_depth: float = 0.1
        self.target_was_visible = False; self.last_size_fraction = 0.0; self.last_at_edge = False
        self.time_unseen = 0.0
    def _visible_target(self, vision: Vision) -> Optional[Tuple[float, float, float]]:
        """(min x, max x, center x) of the target in pixels, or None if not visible."""
        if self.target_type in ('gate', 'either') and vision.is_gate_visible():
            left, right = vision.get_gate_pair()
            return left['min_x'], right['max_x'], vision.get_gate_center_x()
        if self.target_type in ('pole', 'either') and vision.is_pole_visible():
            pole = vision.get_best_pole()
            return pole['min_x'], pole['max_x'], pole['center_x']
        return None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision: Vision, context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading)
        self.target_depth = context.get('target_depth', sensors.depth)
        self.target_was_visible = False; self.last_size_fraction = 0.0; self.last_at_edge = False
        self.time_unseen = 0.0
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        cam_w, cam_h = sensors.camera_image.get_size()
        target = self._visible_target(vision)
        if target is not None:
            min_x, max_x, center_x = target
            margin = self.edge_margin_fraction * cam_w
            self.target_was_visible = True; self.time_unseen = 0.0
            self.last_size_fraction = (max_x - min_x) / cam_w
            self.last_at_edge = min_x < margin or max_x > cam_w - margin
            # Steer at the target: its heading is our heading plus its bearing (positive = left).
            bearing = bearing_deg(CameraModel(cam_w, cam_h, config.cameraFov), center_x)
            self.heading_to_hold = (sensors.heading + bearing) % 360
        else:
            self.time_unseen += dt
            passing = self.last_at_edge or (self.last_size_fraction >= self.pass_size_fraction
                                            and self.time_unseen > self.coast_seconds)
            if self.target_was_visible and passing:
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            if self.time_unseen > self.lost_timeout:
                print(f"ERROR: DriveUntilTargetLost - '{self.target_type}' not seen for {self.lost_timeout}s while still far away")
                return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
        # Drive on while it's in view, or after a far-away miss; wait in place if never seen yet.
        surge = self.surge_power if self.target_was_visible else 0.0
        return SubtaskStatus.RUNNING, sub.get_heading_commands(sensors, self.heading_to_hold, surge, target_depth=self.target_depth)

class WaitForTargetVisible(Subtask):
    """
    Turns in place, holding depth, until the target is visible. When waiting for
    the gate and a single red pole is in view, turns toward that pole, since the
    gate is probably on that side. Otherwise turns the task's search direction
    (1 = right).
    """
    def __init__(self, target_type='either'):
        self.target_type = target_type.lower(); self.target_depth: float = 0.1
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.target_depth = context.get('target_depth', sensors.depth)
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        visible = False
        if self.target_type == 'pole': visible = vision_data.is_pole_visible()
        elif self.target_type == 'gate': visible = vision_data.is_gate_visible()
        else: visible = vision_data.is_pole_visible() or vision_data.is_gate_visible()
        if visible: return SubtaskStatus.COMPLETED, sub.get_spin_damping_commands(sensors, self.target_depth)
        power = context.get('search_turn_power', 0.25)
        yaw = -power * context.get('search_direction', 1)   # Positive yaw turns left
        if self.target_type in ('gate', 'either') and vision_data.red_blobs:
            cam_w, _ = sensors.camera_image.get_size()
            pole = max(vision_data.red_blobs, key=lambda b: b['area'])
            yaw = power if pole['center_x'] < cam_w / 2 else -power
        heave = sub.get_heave_command(sensors, self.target_depth)
        return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0, 0, yaw, heave)

class AlignToObjectX(Subtask):
    def __init__(self, target_x_fraction: float, tolerance_fraction: float = 0.047, yaw_gain: float = 0.8, yaw_rate_tolerance: float = 0.05):
        self.target_x_fraction=target_x_fraction; self.tolerance_fraction=tolerance_fraction; self.yaw_gain=yaw_gain; self.yaw_rate_tolerance=yaw_rate_tolerance
        self.target_depth: float = 0.1
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.target_depth = context.get('target_depth', sensors.depth)
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        current_center_x = None
        if vision_data.is_gate_visible(): current_center_x = vision_data.get_gate_center_x()
        elif vision_data.is_pole_visible(): current_center_x = vision_data.get_pole_center_x()
        if current_center_x is None: return SubtaskStatus.RUNNING, sub.get_spin_damping_commands(sensors, self.target_depth)
        cam_w, _ = sensors.camera_image.get_size(); target_pixel_x = cam_w * self.target_x_fraction
        pixel_error_x = current_center_x - target_pixel_x
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain; yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        is_centered = abs(pixel_error_x) < self.tolerance_fraction * cam_w; is_stable = abs(sensors.imu.gyro_z) < self.yaw_rate_tolerance
        if is_centered and is_stable: context['initial_heading'] = sensors.heading; return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        heave = sub.get_heave_command(sensors, self.target_depth)
        return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw, heave)
    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]): context['initial_heading'] = sensors.heading

class ApproachAndCenterObject(Subtask):
    """Heights are fractions of image height, alignment a fraction of image width."""
    def __init__(self, height_threshold_fraction: float, target_x_fraction: float = 0.5, surge_p_gain: float = 24.0, height_tolerance_fraction: float = 0.02, yaw_gain: float = 1.5, align_tolerance_fraction: float = 0.047, lost_timeout: float = 2.0):
        self.height_threshold = height_threshold_fraction; self.target_x_fraction = target_x_fraction
        self.surge_p_gain = surge_p_gain; self.height_tolerance = height_tolerance_fraction
        self.yaw_gain = yaw_gain; self.align_tolerance_fraction = align_tolerance_fraction
        self.lost_timeout = lost_timeout; self.time_since_target_lost = 0.0; self.target_depth: float = 0.1
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        self.time_since_target_lost = 0.0; self.target_depth = context.get('target_depth', sensors.depth)
        print(f"INFO: Approach starting. Target Height: {self.height_threshold:.2f} of image, Target X: {self.target_x_fraction*100:.0f}%")
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        target_height = vision_data.get_pole_apparent_height(); target_center_x = vision_data.get_pole_center_x(); is_visible = vision_data.is_pole_visible()
        if is_visible and target_center_x is not None:
            self.time_since_target_lost = 0.0
            cam_w, cam_h = sensors.camera_image.get_size(); target_pixel_x = cam_w * self.target_x_fraction
            height_error = self.height_threshold - target_height / cam_h; surge = height_error * self.surge_p_gain; surge = np.clip(surge, -0.4, 0.5)
            pixel_error_x = target_center_x - target_pixel_x
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain; yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
            heave = sub.get_heave_command(sensors, self.target_depth)
            is_at_dist = abs(height_error) < self.height_tolerance; is_aligned = abs(pixel_error_x) < self.align_tolerance_fraction * cam_w
            if is_at_dist and is_aligned: print("INFO: Approach complete."); context['initial_heading'] = sensors.heading; return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, 0.0, yaw, heave)
        else:
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout: print(f"ERROR: Approach failed - target lost for > {self.lost_timeout}s"); return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors, self.target_depth)
            return SubtaskStatus.RUNNING, sub.get_spin_damping_commands(sensors, self.target_depth)
    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]):
        if not self.time_since_target_lost > 0: context['initial_heading'] = sensors.heading