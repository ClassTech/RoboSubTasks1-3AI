#!/usr/bin/env python3
"""
Gymnasium environment for the victory dance: points are scored for each 90 deg
turn in any axis and direction. This sub can only yaw (pitch is passive, and
roll waits for a pool test of how far the side thrusters can roll it), so the
dance is a full turn clockwise with a stop at each 90 deg, then a full turn
back counterclockwise: eight quarter turns. The user asked for it learned, not
programmed. The runner says "dance" right after the gate.

A stop counts once the sub is within STOP_DEG of the stop's heading and nearly
still (under STILL_DPS) for HOLD_S, measured from the heading the dance started
on (so small errors don't add up stop to stop, and the magnetometer's offset
cancels). Judged on the true heading, as the judges see it.

Observation (DanceObservation; measured only, numpy, runs on the Pi), per frame:
  depth, depth rate, pitch, yaw rate
  turned       heading turned since the dance began (unwrapped magnetometer), /360
  deepest      the farthest clockwise turned so far, /360: -180 on the way out and
               on the way back look the same without it
  still        seconds the yaw rate has stayed under STILL_DPS, /HOLD_S
  last action  turn, depth, surge
4 frames stacked.

Reward:
  progress   change in a score: 3 per stop done, minus 1 per 90 deg from the current
             stop's heading, plus 1 x the fraction of HOLD_S held there; minus 1.0 x
             depth error from 1.3 m, minus 0.5 per metre drifted from the start
             (clear of the gate behind and set 1 ahead)
  success    +5 after the eighth stop
  failures   -5 each: collision, surfaced, timeout, lost (more than 180 deg off)
  time       -0.01 per step;  energy  -0.05 per second at full thrust
Starts: as the gate hands over, 1-2.5 m past it near a passage's line, facing on
within 15 deg, 1.2-1.6 m deep, up to 0.3 m/s forward.
"""
import math
import random
from collections import deque
from typing import List, Optional

import numpy as np

from sub_env import SubTaskEnv

STOPS = [-90.0, -180.0, -270.0, -360.0, -270.0, -180.0, -90.0, 0.0]   # Clockwise is negative (heading grows left)
STOP_DEG = 5.0
STILL_DPS = 5.0
HOLD_S = 3.0
TARGET_DEPTH_M = 1.3
FIELDS = ["depth", "depth_rate", "pitch", "yaw_rate", "turned", "deepest", "still",
          "action_turn", "action_depth", "action_surge"]


class DanceObservation:
    """The dance's observation, from the depth sensor, IMU and magnetometer."""
    def __init__(self, stack: int = 4):
        self.stack, self.fields = stack, FIELDS
        self.index = {f: i for i, f in enumerate(FIELDS)}
        self.frame_size = len(FIELDS)
        self.size = self.frame_size * stack
        self.reset()

    def reset(self):
        self.turned, self.deepest, self.still = 0.0, 0.0, 0.0
        self._last_heading = None
        self.frames: deque = deque(maxlen=self.stack)

    def update(self, depth, depth_rate, heading, pitch, yaw_rate, last_action, dt) -> np.ndarray:
        if self._last_heading is not None:
            self.turned += (heading - self._last_heading + 180.0) % 360.0 - 180.0
        self._last_heading = heading
        self.deepest = min(self.deepest, self.turned)
        self.still = self.still + dt if abs(math.degrees(yaw_rate)) < STILL_DPS else 0.0
        frame = np.asarray([depth / 2.0, depth_rate / 0.5, pitch / 20.0, yaw_rate / 1.0,
                            self.turned / 360.0, self.deepest / 360.0, min(self.still, 2 * HOLD_S) / HOLD_S]
                           + [float(v) for v in last_action], dtype=np.float32)
        if not self.frames:
            for _ in range(self.stack - 1):
                self.frames.append(frame)
        self.frames.append(frame)
        return np.concatenate(list(reversed(self.frames)))


class TasksDanceEnv(SubTaskEnv):
    COURSE = "tasks"
    OUTCOMES = ("is_success", "collision", "lost", "surfaced", "timeout")
    EXTRA_INFO = ("stops",)
    ACTION_DIM = 3
    MIN_TARGET_DEPTH_M = 1.0
    MAX_TARGET_DEPTH_M = 1.6

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 120.0,
                 randomize: bool = False, side: Optional[int] = None):
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=0.05, time_penalty=0.01,
                         progress_weight=1.0, timeout_penalty=5.0, randomize=randomize)
        self.builder = DanceObservation()
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    def _place_sub(self, rng: random.Random):
        s, info = self.core.state, self.core.course.info
        s.x = info["gate_x"] + rng.uniform(1.0, 2.5)
        s.y = info["gate_center_y"] + rng.choice((1, -1)) * 0.762 + rng.uniform(-0.4, 0.4)
        s.heading = rng.uniform(-15.0, 15.0) % 360
        s.z = rng.uniform(1.2, 1.6)
        speed = rng.uniform(0.0, 0.3)
        s.velocity_x, s.velocity_y = speed * math.cos(math.radians(s.heading)), speed * math.sin(math.radians(s.heading))

    def _on_reset(self):
        s = self.core.state
        self.start_xy, self.start_heading = (s.x, s.y), s.heading
        self.true_turned, self._last_true = 0.0, s.heading
        self.stop, self.held = 0, 0.0
        self.builder.reset()

    def _err_deg(self) -> float:
        return self.true_turned - STOPS[min(self.stop, len(STOPS) - 1)]

    def _physics_outcome(self) -> Optional[str]:
        s, dt = self.core.state, self.core.config.physicsDt
        self.true_turned += (s.heading - self._last_true + 180.0) % 360.0 - 180.0
        self._last_true = s.heading
        if self.stop >= len(STOPS):
            return None
        err = self._err_deg()
        if abs(err) > 180.0:
            return "lost"
        if abs(err) < STOP_DEG and abs(math.degrees(s.angular_velocity_z)) < STILL_DPS:
            self.held += dt
            if self.held >= HOLD_S:
                self.stop, self.held = self.stop + 1, 0.0
        else:
            self.held = 0.0
        return None

    def done_label(self) -> bool:
        return self.stop >= len(STOPS)

    def _complete(self) -> bool:
        return self.done_label()

    def _terminal_reward(self, outcome: str) -> float:
        return 5.0 if outcome == "success" else -5.0

    def _potential(self) -> float:
        s = self.core.state
        drift = math.hypot(s.x - self.start_xy[0], s.y - self.start_xy[1])
        if self.stop >= len(STOPS):
            stops_part = 3.0 * len(STOPS)
        else:
            stops_part = 3.0 * self.stop - abs(self._err_deg()) / 90.0 + self.held / HOLD_S
        return stops_part - 1.0 * abs(s.z - TARGET_DEPTH_M) - 0.5 * drift

    def _observe(self) -> np.ndarray:
        r = self.readings
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_z,
                                  np.asarray(self.last_action[:3]), self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _episode_info(self) -> dict:
        return {"stops": self.stop}

    def scripted_mission(self) -> List:
        """No scripted dance (it's learned only); the baseline just holds station, for train.py's comparison."""
        from ai.tasks.stabilize_task import StabilizeTask
        return [StabilizeTask(duration=self.max_seconds, target_depth=TARGET_DEPTH_M)]
