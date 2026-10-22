#!/usr/bin/env python3
"""
The sub's low-level control loops: depth (PID), pitch (PD) and heading (PD),
plus the mixer that turns surge/sway/yaw/heave/pitch efforts into the six
thruster commands. Shared by the scripted tasks, the RL environment and the
Raspberry Pi (numpy + standard library only).

Only measured values are used: depth and depth rate from the pressure sensor,
heading from the magnetometer, pitch and rates from the IMU.
"""
from dataclasses import dataclass

import numpy as np

from core_types import ThrusterCommands


def angle_diff(a1: float, a2: float) -> float:
    """Shortest difference a1 - a2 between two angles in degrees, in (-180, 180]."""
    d = (a1 - a2) % 360
    return d - 360 if d > 180 else d


@dataclass
class ControllerGains:
    # Depth gains retuned for the measured thrust (3.56 N per thruster) and the draggy frame: settles a
    # 0.5 m step in ~4.4 s with ~3 cm overshoot. (The old 2.5 / 0.4 / 5.0 took ~18 s and overshot 19 cm:
    # the integral wound up during the slow vertical transit.)
    depth_p: float = 9.0
    depth_i: float = 0.1
    depth_d: float = 3.0
    depth_i_clamp: float = 2.0
    heading_p: float = 0.1      # Per degree of heading error
    yaw_d: float = 4.0          # Per rad/s of yaw rate


def depth_effort(gains: ControllerGains, depth: float, depth_rate: float, depth_integral: float,
                 target_depth: float) -> float:
    """Heave effort in [-1, 1]. depth_integral is the accumulated depth error (m s)."""
    heave = (target_depth - depth) * gains.depth_p + depth_integral * gains.depth_i - depth_rate * gains.depth_d
    return float(np.clip(heave, -1.0, 1.0))


def yaw_effort(gains: ControllerGains, heading: float, yaw_rate: float, target_heading: float) -> float:
    return float(np.clip(angle_diff(target_heading, heading) * gains.heading_p - yaw_rate * gains.yaw_d, -1.0, 1.0))


def mix(surge: float, sway: float, yaw: float, heave: float) -> ThrusterCommands:
    """
    Mixes 4-axis efforts into 6 thruster commands. The two vertical thrusters are
    on the sides, so they only move the sub up and down (pitch is passively
    stable, not controlled). The horizontal group is scaled down on its own if
    any of its commands exceeds 1. (Scaling all six together meant hard turning
    and sliding cut the vertical thrust, and the buoyant sub floated up.)
    """
    cos_45 = 0.7071
    surge_cmd, sway_cmd = surge * cos_45, sway * cos_45
    c = ThrusterCommands(
        hfl=surge_cmd + sway_cmd + yaw, hfr=surge_cmd - sway_cmd - yaw,
        hal=surge_cmd - sway_cmd + yaw, har=surge_cmd + sway_cmd - yaw,
        vl=heave, vr=heave)
    h_max = max(1.0, abs(c.hfl), abs(c.hfr), abs(c.hal), abs(c.har))
    c.hfl /= h_max; c.hfr /= h_max; c.hal /= h_max; c.har /= h_max
    return c


class SetpointController:
    """
    Holds a heading and depth at given surge and sway efforts. Used by the policy (in sim and on the Pi).

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
               surge: float, sway: float = 0.0) -> ThrusterCommands:
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
        yaw = yaw_effort(g, heading, yaw_rate, target_heading)
        return mix(float(np.clip(surge, -1.0, 1.0)), float(np.clip(sway, -1.0, 1.0)), yaw, heave)
