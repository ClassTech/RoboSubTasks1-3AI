#!/usr/bin/env python3
"""
Simulated versions of the real sub's sensors: pressure (depth), magnetometer
(heading) and IMU (pitch, rates, acceleration). Pure math (no pygame) so it can
be shared by the graphical simulator and a headless training environment.

Each reset() draws new per-episode errors (offsets and biases), so a policy
can't learn to rely on one particular sensor calibration.
"""
import math
import random
from dataclasses import dataclass
from typing import Optional

from config import SensorNoiseConfig
from core_types import MPU6050Readings
from world import SubmarinePhysicsState


@dataclass
class SensorReadings:
    depth: float          # m, from the pressure sensor
    depth_rate: float     # m/s, positive = going deeper, from the depth filter
    heading: float        # deg [0, 360), from the magnetometer
    pitch: float          # deg, from the IMU
    imu: MPU6050Readings  # Rates (rad/s) and accelerations (m/s^2), with noise


class SensorModel:
    def __init__(self, noise: Optional[SensorNoiseConfig] = None, rng: Optional[random.Random] = None):
        self.noise = noise or SensorNoiseConfig()
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        n, r = self.noise, self.rng
        self.depth_offset = r.uniform(*n.depthOffsetRange)
        self.heading_offset = r.uniform(*n.headingOffsetRange)
        self.gyro_bias_z = r.uniform(*n.gyroBiasRange)
        self.gyro_bias_y = r.uniform(*n.gyroBiasRange)
        self._depth_est: Optional[float] = None
        self._depth_rate_est = 0.0

    def _filter_depth(self, measured: float, dt: float):
        """Alpha-beta filter: smoothed depth plus a depth rate estimate."""
        if self._depth_est is None or dt <= 0:
            self._depth_est = measured
            return
        predicted = self._depth_est + self._depth_rate_est * dt
        residual = measured - predicted
        self._depth_est = predicted + self.noise.depthFilterAlpha * residual
        self._depth_rate_est += (self.noise.depthFilterBeta / dt) * residual

    def read(self, state: SubmarinePhysicsState, true_imu: MPU6050Readings, dt: float) -> SensorReadings:
        n, r = self.noise, self.rng
        raw_depth = state.z + self.depth_offset + r.gauss(0.0, n.depthNoiseStd)
        self._filter_depth(raw_depth, dt)

        imu = MPU6050Readings(
            accel_x=true_imu.accel_x + r.gauss(0.0, n.accelNoiseStd),
            accel_y=true_imu.accel_y + r.gauss(0.0, n.accelNoiseStd),
            accel_z=true_imu.accel_z + r.gauss(0.0, n.accelNoiseStd),
            gyro_z=true_imu.gyro_z + self.gyro_bias_z + r.gauss(0.0, n.gyroNoiseStd),
            gyro_y=true_imu.gyro_y + self.gyro_bias_y + r.gauss(0.0, n.gyroNoiseStd),
        )
        return SensorReadings(
            depth=raw_depth,
            depth_rate=self._depth_rate_est,
            heading=(state.heading + self.heading_offset + r.gauss(0.0, n.headingNoiseStd)) % 360,
            pitch=state.pitch + r.gauss(0.0, n.pitchNoiseStd),
            imu=imu,
        )
