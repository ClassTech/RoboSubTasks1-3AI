#!/usr/bin/env python3
"""
The simulation without any display: the pool course, the sub's physics at a
fixed time step, collisions, gate passes, and the simulated sensors and
detector. No pygame or OpenCV, so it can run many copies in parallel for
training. The graphical SubmarineSimulator draws on top of this.
"""
import math
import random
from typing import Callable, Optional, Tuple

import numpy as np

from collision import build_obstacles, find_collision, check_gate_crossing, check_pass
from config import SimulationConfig, PrequalConfig, TasksConfig
from course import Course, prequal_course, tasks_course
from core_types import MPU6050Readings, ThrusterCommands
from detector_model import DetectorModel, Detections
from perception import CameraModel
from sensor_model import SensorModel, SensorReadings
from world import PrequalGate, PrequalMarker, SubmarinePhysicsState


class SubSim:
    def __init__(self, config: Optional[SimulationConfig] = None,
                 prequal: Optional[PrequalConfig] = None,
                 detector_mode: Optional[str] = "noisy",
                 course: str = "prequal", tasks: Optional[TasksConfig] = None,
                 rng: Optional[random.Random] = None,
                 log: Optional[Callable[[str], None]] = print):
        """
        detector_mode: 'noisy', 'perfect', or None (no simulated detector; the
        caller renders the camera and runs OpenCV itself).
        rng: source of all randomness. Defaults to one seeded from the global
        random module, so random.seed() makes runs repeatable.
        log: where event messages (gate passes, collisions) go; None to silence.
        course: 'prequal' (the qualification course) or 'tasks' (RoboSub tasks 1-3); see course.py.
        """
        assert detector_mode in ("noisy", "perfect", None), detector_mode
        assert course in ("prequal", "tasks"), course
        self.course_kind = course
        self.tasks = tasks or TasksConfig()
        self.config = config or SimulationConfig()
        self.prequal = prequal or PrequalConfig()
        self.detector_mode = detector_mode
        self.rng = rng or random.Random(random.random())
        self.log = log or (lambda msg: None)

        c = self.config
        self.camera = CameraModel(c.cameraWidth, c.cameraHeight, c.cameraFov)
        self.pole_bottom_z = c.worldDepth - 0.01
        self.net_buoyancy = c.subVolume * c.waterDensity * c.gravity - c.subMass * c.gravity
        # Corner thrusters push at right angles to the line from the center,
        # so their yaw lever arm is their distance from the center.
        self.yaw_lever_arm = c.thrusterArm

        self.sensors = SensorModel(rng=random.Random(self.rng.random()))
        self.reset()

    def reset(self):
        c, p = self.config, self.prequal
        if self.course_kind == "prequal":
            self.course: Course = prequal_course(c, p, self.pole_bottom_z)
            # The qualification code (its environments and scripted tasks) uses these directly
            self.gate = PrequalGate(
                x=p.GATE_X_POS, center_y=c.worldHeight / 2,
                z_top=p.GATE_DEPTH_METERS, width=p.GATE_WIDTH_METERS,
                height=p.GATE_OPENING_HEIGHT, color=p.GATE_COLOR)
            self.marker = PrequalMarker(
                x=p.MARKER_X_POS, y=c.worldHeight / 2,
                z_top=-p.POLE_ABOVE_SURFACE_METERS, z_bottom=self.pole_bottom_z,
                radius=p.MARKER_DIAMETER_METERS / 2, color=p.MARKER_COLOR)
        else:
            self.course = tasks_course(c, self.tasks, self.pole_bottom_z, self.rng)
            self.gate = self.marker = None
        self.segments = self.course.segments
        self.obstacles = build_obstacles(self.segments)
        self.detector = None
        if self.detector_mode:
            self.detector = DetectorModel(self.camera, self.segments,
                                          noisy=self.detector_mode == "noisy",
                                          rng=random.Random(self.rng.random()))
        st = self.course.start
        # (The qualification start's sideways offset is drawn here, after the detector, as it always was)
        y = st["y"] + (self.rng.uniform(-0.5, 0.5) if self.course_kind == "prequal" else 0.0)
        self.state = SubmarinePhysicsState(x=st["x"], y=y, z=st["z"], heading=st["heading"], pitch=0.0)
        self.true_imu = MPU6050Readings()
        self.thrust = ThrusterCommands()   # What the thrusters are actually doing (lags the commands)
        self.sensors.reset()
        self.time = 0.0
        self._accumulator = 0.0
        self.gate_passes = 0
        self.crossings = []          # Qualification course: every GateCrossing of the gate plane, in order
        self.passes = []             # Every crossing of any pass line (collision.Pass), in order
        self.collision_count = 0
        self.last_collision = None
        self._time_since_contact = float('inf')

    # --- Sensing ---
    def sense(self, dt: float) -> Tuple[SensorReadings, Optional[Detections]]:
        """What the sub's sensors and (simulated) detector report now. dt: time since last call."""
        detections = self.detect(dt)
        return self.read_sensors(dt), detections

    def read_sensors(self, dt: float) -> SensorReadings:
        """Pressure sensor, magnetometer and IMU readings. dt: time since the last reading."""
        return self.sensors.read(self.state, self.true_imu, dt)

    def detect(self, dt: float) -> Optional[Detections]:
        """The simulated detector's boxes, or None if there is none. dt: time since the last call."""
        return self.detector.detect(self.state, dt) if self.detector else None

    # --- Time ---
    def advance(self, dt: float, commands: ThrusterCommands):
        """Advances the world by dt seconds, in fixed physics steps, holding the commands."""
        self._accumulator += dt
        step = self.config.physicsDt
        while self._accumulator >= step - 1e-9:
            self._physics_step(step, commands)
            self._accumulator -= step

    def _ramp(self, commands: ThrusterCommands, dt: float) -> ThrusterCommands:
        """The sub's software limits how fast each thruster's command can change."""
        step = dt / self.config.thrusterRampSec if self.config.thrusterRampSec > 0 else float("inf")
        t = self.thrust
        for name in ("hfl", "hfr", "hal", "har", "vl", "vr"):
            cur, want = getattr(t, name), getattr(commands, name)
            setattr(t, name, cur + max(-step, min(step, want - cur)))
        return t

    def _physics_step(self, dt: float, commands: ThrusterCommands):
        c, s = self.config, self.state
        commands = self._ramp(commands, dt)
        f_max = c.thrusterMaxForce
        f_hfl, f_hfr = commands.hfl * f_max, commands.hfr * f_max
        f_hal, f_har = commands.hal * f_max, commands.har * f_max
        f_vl, f_vr = commands.vl * f_max, commands.vr * f_max
        cos_45 = 0.7071
        thrust_surge = (f_hfl + f_hfr + f_hal + f_har) * cos_45
        thrust_sway = (-f_hfl + f_hfr + f_hal - f_har) * cos_45
        thrust_heave = f_vl + f_vr   # The vertical pair is on the sides: no pitch torque (roll isn't modeled)
        thrust_yaw = (-f_hfl + f_hfr - f_hal + f_har) * self.yaw_lever_arm
        # Pitch is passive: weight below buoyancy rights the sub (it drives flat)
        righting_pitch = -c.pitchRightingMoment * math.sin(math.radians(s.pitch))

        h_rad, p_rad = math.radians(s.heading), math.radians(s.pitch)
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
        cos_p, sin_p = math.cos(p_rad), math.sin(p_rad)
        vel_surge = s.velocity_x * cos_h + s.velocity_y * sin_h
        vel_sway = -s.velocity_x * sin_h + s.velocity_y * cos_h
        vel_heave = s.velocity_z
        drag_surge = -c.surgeDragCoeff * vel_surge * abs(vel_surge)
        drag_sway = -c.swayDragCoeff * vel_sway * abs(vel_sway)
        drag_heave = -c.heaveDragCoeff * vel_heave * abs(vel_heave)
        drag_yaw = -c.angularDragCoeff_Z * s.angular_velocity_z ** 2 * np.sign(s.angular_velocity_z)
        drag_pitch = -c.angularDragCoeff_Y * s.angular_velocity_y ** 2 * np.sign(s.angular_velocity_y)

        force_surge = thrust_surge + drag_surge
        force_sway = thrust_sway + drag_sway
        force_heave = thrust_heave + drag_heave
        f_x_pitched = force_surge * cos_p - force_heave * sin_p
        f_z_pitched = force_surge * sin_p + force_heave * cos_p
        ax = (f_x_pitched * cos_h - force_sway * sin_h) / c.subMass
        ay = (f_x_pitched * sin_h + force_sway * cos_h) / c.subMass
        az = (f_z_pitched - self.net_buoyancy) / c.subMass

        s.velocity_x += ax * dt
        s.velocity_y += ay * dt
        s.velocity_z += az * dt
        s.angular_velocity_z += (thrust_yaw + drag_yaw) / c.subInertia_Z * dt
        s.angular_velocity_y += (righting_pitch + drag_pitch) / c.subInertia_Y * dt
        self.true_imu = MPU6050Readings(
            accel_x=-ax * sin_h + ay * cos_h,   # Sway
            accel_y=ax * cos_h + ay * sin_h,    # Surge
            accel_z=az,                         # Heave
            gyro_z=s.angular_velocity_z,
            gyro_y=s.angular_velocity_y)

        prev = (s.x, s.y, s.z, s.heading, s.pitch)
        s.x += s.velocity_x * dt
        s.y += s.velocity_y * dt
        s.z += s.velocity_z * dt
        s.heading = (s.heading + math.degrees(s.angular_velocity_z * dt)) % 360
        s.pitch = float(np.clip(s.pitch + math.degrees(s.angular_velocity_y * dt), -90, 90))
        margin = 0.5
        s.x = float(np.clip(s.x, margin, c.worldWidth - margin))
        s.y = float(np.clip(s.y, margin, c.worldHeight - margin))
        s.z = float(np.clip(s.z, 0.0, c.worldDepth - 0.2))
        self.time += dt

        self._resolve_collisions(prev, dt)
        for line in self.course.pass_lines:
            ps = check_pass(line, prev[0], prev[1], prev[2], s.x, s.y, s.z)
            if ps:
                self.passes.append(ps)
                if self.course_kind == "tasks":
                    way = "forward" if ps.direction > 0 else "back"
                    self.log(f"{ps.line.capitalize()} crossed {way}: " + (f"{ps.lane} lane" if ps.lane else "MISSED")
                             + f" (y={ps.y:.2f}, z={ps.z:.2f})")
        crossing = check_gate_crossing(self.gate, prev[0], prev[1], prev[2], s.x, s.y, s.z,
                                       self.pole_bottom_z) if self.gate else None
        if crossing:
            self.crossings.append(crossing)
            if crossing.inside_opening:
                self.gate_passes += 1
            where = "through the gate" if crossing.inside_opening else "outside the gate opening"
            self.log(f"Gate plane crossed {'forward' if crossing.direction > 0 else 'backward'} {where} "
                     f"(y={crossing.y:.2f}, z={crossing.z:.2f})")

    def _resolve_collisions(self, prev, dt: float):
        """On contact, return the sub to its last clear pose and stop it."""
        c, s = self.config, self.state
        hit = find_collision(s.x, s.y, s.z, s.heading, c.submarineLength,
                             c.submarineWidth, c.submarineHeight, self.obstacles)
        clear_for = self._time_since_contact
        self._time_since_contact = 0.0 if hit else self._time_since_contact + dt
        if hit:
            # A new collision, unless it's the same contact bumping again within half a second
            if clear_for > 0.5:
                self.collision_count += 1
                self.log(f"Collision with {hit.name} at x={s.x:.2f}, y={s.y:.2f}, z={s.z:.2f}")
            s.x, s.y, s.z, s.heading, s.pitch = prev
            s.velocity_x = s.velocity_y = s.velocity_z = 0.0
            s.angular_velocity_z = s.angular_velocity_y = 0.0
        self.last_collision = hit
