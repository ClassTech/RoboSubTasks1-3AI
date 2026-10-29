#!/usr/bin/env python3
"""
The sub's low-level control loops: depth (PID), pitch (PD) and heading (PD),
plus the mixer that turns surge/sway/yaw/heave/pitch efforts into the six
thruster commands. Shared by the scripted tasks, the RL environment and the
Raspberry Pi (numpy + standard library only).

Only measured values are used: depth and depth rate from the pressure sensor,
heading from the magnetometer, pitch and rates from the IMU.
"""
import math
from dataclasses import dataclass

import numpy as np

from core_types import ThrusterCommands

ROLL_HEAVE_DEG = 20.0   # Rolled further than this, depth is held with the corner thrusters too


def angle_diff(a1: float, a2: float) -> float:
    """Shortest difference a1 - a2 between two angles in degrees, in (-180, 180]."""
    d = (a1 - a2) % 360
    return d - 360 if d > 180 else d


@dataclass
class ControllerGains:
    # Depth gains tuned for the measured thrust (7.85 N per thruster) and the draggy frame: settle a
    # 0.5 m step in ~1.8 s with ~1 cm overshoot, and ~2.1 s at the randomized extremes. (9.0 / 0.1 / 3.0,
    # tuned when the thrust was taken as 3.56 N, rang for ~20 s at the real thrust against the 0.5 s ramp.)
    depth_p: float = 4.5
    depth_i: float = 0.05
    depth_d: float = 1.0
    depth_i_clamp: float = 2.0
    # Heading gains set at 3.56 N. At the real thrust they wobble up to 14 deg/s holding a heading (on the
    # magnetometer's noise), against ~6 at 3.56 N; halved (0.05 / 2.0) they're back to ~6, turning as fast.
    # The skills were trained on these and lost 11/180 in the mission on the halved ones (outside their
    # +-30% training spread), so only the dance, which holds still at its stops, halves them.
    heading_p: float = 0.1      # Per degree of heading error
    yaw_d: float = 4.0          # Per rad/s of yaw rate


def depth_effort(gains: ControllerGains, depth: float, depth_rate: float, depth_integral: float,
                 target_depth: float) -> float:
    """Heave effort in [-1, 1]. depth_integral is the accumulated depth error (m s)."""
    heave = (target_depth - depth) * gains.depth_p + depth_integral * gains.depth_i - depth_rate * gains.depth_d
    return float(np.clip(heave, -1.0, 1.0))


def yaw_effort(gains: ControllerGains, heading: float, yaw_rate: float, target_heading: float) -> float:
    return float(np.clip(angle_diff(target_heading, heading) * gains.heading_p - yaw_rate * gains.yaw_d, -1.0, 1.0))


def mix(surge: float, sway: float, yaw: float, heave: float, roll: float = 0.0) -> ThrusterCommands:
    """
    Mixes 5-axis efforts into 6 thruster commands. The two vertical thrusters are
    on the sides, so together they move the sub up and down and apart they roll
    it (positive roll: right side down); pitch is passively stable, not
    controlled. Roll has first call on the vertical pair; heave is clipped to
    what's left (with no roll, as before: heave alone, within 1). The horizontal group is scaled down on its own if
    any of its commands exceeds 1. (Scaling all six together meant hard turning
    and sliding cut the vertical thrust, and the buoyant sub floated up.)

    The corner thrusters are angled 45 deg across the diagonals, so each pushes
    toward the far side and turns the sub away from its own side: front-left
    running forward pushes forward-right and turns right. Positive sway and yaw
    are left. (Before 2026-10-24 the sim and this mixer had every horizontal
    thruster mirrored left for right; consistent with each other, so nothing
    flew differently, but the names didn't match the real sub's thrusters.)
    """
    cos_45 = 0.7071
    surge_cmd, sway_cmd = surge * cos_45, sway * cos_45
    # Roll comes first in the vertical pair: heave gets what's left. (Scaled together, a saturated depth
    # loop halved the roll torque, below the righting moment, and the sub stalled partway round.)
    room = 1.0 - min(1.0, abs(roll))
    heave = max(-room, min(room, heave))
    c = ThrusterCommands(
        hfl=surge_cmd - sway_cmd - yaw, hfr=surge_cmd + sway_cmd + yaw,
        hal=surge_cmd + sway_cmd - yaw, har=surge_cmd - sway_cmd + yaw,
        vl=heave - roll, vr=heave + roll)
    h_max = max(1.0, abs(c.hfl), abs(c.hfr), abs(c.hal), abs(c.har))
    c.hfl /= h_max; c.hfr /= h_max; c.hal /= h_max; c.har /= h_max
    return c


class SetpointController:
    """
    Holds a heading and depth at given surge, sway and roll efforts (roll open loop: the sub rights itself). Used by the policy (in sim and on the Pi).

    The targets move toward the commanded ones at limited rates: depth at
    max_depth_target_rate m/s, heading at max_heading_target_rate deg/s. A policy
    that flips its commands from step to step otherwise slams the thrusters
    between extremes; with their 0.5 s ramp that set the pitch swinging and lost
    depth, and made the sub wag in yaw. Steady turns and depth changes still go
    through, just no faster than these rates.
    """
    def __init__(self, gains: ControllerGains = None, max_depth_target_rate: float = 0.5,
                 max_heading_target_rate: float = 30.0):
        self.gains = gains or ControllerGains()
        self.max_depth_target_rate = max_depth_target_rate
        self.max_heading_target_rate = max_heading_target_rate
        self.reset()

    def reset(self):
        self.depth_integral = 0.0
        self.depth_target = None     # The rate-limited targets actually being held
        self.heading_target = None

    def update(self, dt: float, depth: float, depth_rate: float, heading: float, pitch: float,
               pitch_rate: float, yaw_rate: float, target_heading: float, target_depth: float,
               surge: float, sway: float = 0.0, roll: float = 0.0, roll_angle: float = 0.0) -> ThrusterCommands:
        """roll: open-loop roll effort. roll_angle: the IMU's roll (deg). Rolled past ROLL_HEAVE_DEG the depth
        loop's push is split between the body's axes: heave x cos(roll) from the vertical pair, and sway x
        -sin(roll) from the 45 deg corner thrusters, whose sideways push points up or down when the sub is
        on its side (user). Full sway and full heave are both 2 x the thrust (15.7 N), so the efforts need
        no scaling. (Heave alone drove a rolling sub to the surface: on its side it pushed sideways, and
        upside down the wrong way, while roll took the vertical pair.) Level flight is unchanged."""
        g = self.gains
        if self.depth_target is None:
            self.depth_target = target_depth
        step = self.max_depth_target_rate * dt
        self.depth_target += float(np.clip(target_depth - self.depth_target, -step, step))
        target_depth = self.depth_target
        if self.heading_target is None:
            self.heading_target = heading
        turn_step = self.max_heading_target_rate * dt
        self.heading_target = (self.heading_target
                               + float(np.clip(angle_diff(target_heading, self.heading_target), -turn_step, turn_step))) % 360
        target_heading = self.heading_target
        self.depth_integral = float(np.clip(self.depth_integral + (target_depth - depth) * dt,
                                            -g.depth_i_clamp, g.depth_i_clamp))
        heave = depth_effort(g, depth, depth_rate, self.depth_integral, target_depth)
        if abs(roll_angle) > ROLL_HEAVE_DEG:
            r = math.radians(roll_angle)
            heave, sway = heave * math.cos(r), sway - heave * math.sin(r)
        yaw = yaw_effort(g, heading, yaw_rate, target_heading)
        return mix(float(np.clip(surge, -1.0, 1.0)), float(np.clip(sway, -1.0, 1.0)), yaw, heave,
                   float(np.clip(roll, -1.0, 1.0)))
