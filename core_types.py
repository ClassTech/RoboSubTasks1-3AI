#!/usr/bin/env python3
"""
Plain data types shared by the simulation core and the sub's control code.
No pygame or OpenCV, so the headless simulator can import them.
"""
from dataclasses import dataclass


@dataclass
class MPU6050Readings:
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_y: float = 0.0  # Pitch rate
    gyro_z: float = 0.0  # Yaw rate
    gyro_x: float = 0.0  # Roll rate


@dataclass
class ThrusterCommands:
    # Horizontal Vectored Thrusters (4 corners)
    hfl: float = 0.0  # Horizontal Front-Left
    hfr: float = 0.0  # Horizontal Front-Right
    hal: float = 0.0  # Horizontal Aft-Left
    har: float = 0.0  # Horizontal Aft-Right

    # Vertical Thrusters, one on each side: together they move the sub up and down, and
    # apart they roll it. (They can't control pitch; the sub is passively stable in pitch and roll.)
    vl: float = 0.0   # Vertical Left
    vr: float = 0.0   # Vertical Right

    pause_simulation: bool = False
