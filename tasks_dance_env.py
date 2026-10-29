#!/usr/bin/env python3
"""
Gymnasium environment for the victory dance: points are scored for each 90 deg
turn in any axis and direction. The user asked for it learned, not programmed.
The runner says "dance" right after the gate.

Yaw, then roll (RoboSim's order):
  yaw    a full turn clockwise with a stop at each 90 deg, then a full turn back
         counterclockwise: eight quarter turns. A stop counts once the sub is within
         STOP_DEG of the stop's heading and nearly still (under STILL_DPS) for HOLD_S,
         measured from the heading the dance started on (so small errors don't add up
         stop to stop, and the magnetometer's offset cancels).
  roll   a full roll right (right side down), then a full roll back left: eight more
         quarters, each counted on reaching it (within ROLL_PASS_DEG) with no hold: rolled
         past ~60 deg the side thrusters can't hold the sub there, and on its side or
         upside down they push it sideways and the buoyant sub rises. The last quarter,
         level again, is held still for LEVEL_HOLD_S.
Judged on the true heading and roll, as the judges see them. Roll's numbers are
guesses (config.py: nothing measured), so training varies the righting moment widely.

Actions (4), at 10 Hz:
  turn   where to point, measured from the dance's starting heading: -1..1 -> -400..+40
         deg (clockwise negative). The controller steps toward it at most 45 deg per
         action, the long way round when that's the way to it. (The first dance turned
         relative to the current heading each step, like the other skills: holding a
         stop for 3 s then needed a near-zero action 30 steps running, and with any
         exploration it never counted one.)
  depth  target depth, 1.0-1.8 m
  surge  open-loop forward/back thrust
  roll   open-loop roll effort: the side vertical thrusters pushed apart (+ = right side down)

Observation (DanceObservation; measured only, numpy, runs on the Pi), per frame:
  depth, depth rate, pitch, yaw rate
  turned       heading turned since the dance began (unwrapped magnetometer), /360
  deepest      the farthest clockwise turned so far, /360: -180 on the way out and
               on the way back look the same without it
  still        seconds the yaw rate has stayed under STILL_DPS, /HOLD_S
  roll         sin and cos of the IMU's roll, and the roll rate
  rolled       roll since the dance began (unwrapped), /360, and the farthest so far
  level        seconds level and still in roll, /LEVEL_HOLD_S
  last action  turn, depth, surge, roll
4 frames stacked.

Reward:
  progress   change in a score: 3 per stop or quarter done, minus 1 per 90 deg from the
             current one, plus 1 x the fraction of its hold done; minus 1 per 90 deg of
             roll during the yaw part and of heading off the start during the roll part;
             minus 1.0 x depth error from 1.3 m, minus 0.5 per metre drifted from the start
             (clear of the gate behind and set 1 ahead)
  success    +5 after the last quarter
  failures   -5 each: collision, surfaced, timeout, lost (more than 180 deg off)
  time       -0.01 per step;  energy  -0.05 per second at full thrust
Starts: as the gate hands over, 1-2.5 m past it near a passage's line, facing on
within 15 deg, 1.2-1.6 m deep, up to 0.3 m/s forward. In training, roll_share of
episodes start at the roll part (the yaw part done), so it gets practised from the start.
"""
import math
import random
from collections import deque
from typing import List, Optional

import numpy as np

from controller import angle_diff
from sub_env import HEADING_CHANGE_DEG, SubTaskEnv

YAW_STOPS = [-90.0, -180.0, -270.0, -360.0, -270.0, -180.0, -90.0, 0.0]   # Clockwise is negative (heading grows left)
ROLL_STOPS = [90.0, 180.0, 270.0, 360.0, 270.0, 180.0, 90.0, 0.0]          # Right side down is positive
STAGES = [("yaw", a) for a in YAW_STOPS] + [("roll", a) for a in ROLL_STOPS]
STOP_DEG = 5.0
STILL_DPS = 5.0
HOLD_S = 3.0
ROLL_PASS_DEG = 15.0
LEVEL_HOLD_S = 2.0
TURN_RANGE_DEG = (-400.0, 40.0)
TARGET_DEPTH_M = 1.3
FIELDS = ["depth", "depth_rate", "pitch", "yaw_rate", "turned", "deepest", "still",
          "roll_sin", "roll_cos", "roll_rate", "rolled", "rolled_most", "level",
          "action_turn", "action_depth", "action_surge", "action_roll"]


class DanceObservation:
    """The dance's observation, from the depth sensor, IMU and magnetometer."""
    def __init__(self, stack: int = 4):
        self.stack, self.fields = stack, FIELDS
        self.index = {f: i for i, f in enumerate(FIELDS)}
        self.frame_size = len(FIELDS)
        self.size = self.frame_size * stack
        self.reset()

    def reset(self, yaw_done: bool = False):
        """yaw_done: start at the roll part, as if the yaw part had just finished."""
        self.turned, self.deepest, self.still = 0.0, (-360.0 if yaw_done else 0.0), 0.0
        self.rolled, self.rolled_most, self.level = 0.0, 0.0, 0.0
        self._last_heading = self._last_roll = None
        self.frames: deque = deque(maxlen=self.stack)

    def update(self, depth, depth_rate, heading, pitch, yaw_rate, roll, roll_rate, last_action, dt) -> np.ndarray:
        if self._last_heading is not None:
            self.turned += angle_diff(heading, self._last_heading)
            self.rolled += angle_diff(roll, self._last_roll)
        self._last_heading, self._last_roll = heading, roll
        self.deepest = min(self.deepest, self.turned)
        self.rolled_most = max(self.rolled_most, self.rolled)
        self.still = self.still + dt if abs(math.degrees(yaw_rate)) < STILL_DPS else 0.0
        level = abs(roll) < STOP_DEG and abs(math.degrees(roll_rate)) < 2 * STILL_DPS
        self.level = self.level + dt if level else 0.0
        r = math.radians(roll)
        frame = np.asarray([depth / 2.0, depth_rate / 0.5, pitch / 20.0, yaw_rate / 1.0,
                            self.turned / 360.0, self.deepest / 360.0, min(self.still, 2 * HOLD_S) / HOLD_S,
                            math.sin(r), math.cos(r), roll_rate / 3.0, self.rolled / 360.0,
                            self.rolled_most / 360.0, min(self.level, 2 * LEVEL_HOLD_S) / LEVEL_HOLD_S]
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
    ACTION_DIM = 4
    MIN_TARGET_DEPTH_M = 1.0
    MAX_TARGET_DEPTH_M = 1.8   # Deep before rolling: the rolls use all the vertical thrust, and the sub rises ~0.1 m/s
    RIGHTING_SCALE = (0.5, 1.5)   # Training: the roll righting moment is a guess, so vary it widely
    HEADING_GAIN_SCALE = 0.5      # The heading gains halved: at the real thrust the defaults wobble up to 14 deg/s
                                  # holding a heading, past the stops' STILL_DPS (controller.ControllerGains)

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 120.0,
                 randomize: bool = False, side: Optional[int] = None, roll_share: float = 0.0):
        """roll_share: share of episodes that start at the roll part (training)."""
        self.roll_share = roll_share
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
        self._roll_start = rng.random() < self.roll_share
        if self.randomize:
            self.core.config.rollRightingMoment *= rng.uniform(*self.RIGHTING_SCALE)

    def _vary_dynamics(self, rng: random.Random):
        super()._vary_dynamics(rng)
        g = self.controller.gains
        g.heading_p *= self.HEADING_GAIN_SCALE
        g.yaw_d *= self.HEADING_GAIN_SCALE

    def _on_reset(self):
        s = self.core.state
        self.start_xy, self.start_heading = (s.x, s.y), s.heading
        self.true_turned, self._last_true = 0.0, s.heading
        self.true_rolled, self._last_roll = 0.0, s.roll
        self.stage = len(YAW_STOPS) if self._roll_start else 0
        self.held = 0.0
        self.builder.reset(yaw_done=self._roll_start)

    # --- Acting ---
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        lo, hi = TURN_RANGE_DEG
        want = lo + (float(a[0]) + 1) / 2 * (hi - lo)
        turn = float(np.clip(want - self.builder.turned, -HEADING_CHANGE_DEG, HEADING_CHANGE_DEG))
        self.target_heading = (self.readings.heading + turn) % 360
        self.target_depth = self.MIN_TARGET_DEPTH_M + (float(a[1]) + 1) / 2 * (self.MAX_TARGET_DEPTH_M - self.MIN_TARGET_DEPTH_M)
        surge, roll = float(a[2]), float(a[3])

        def driver(r, detections, dt: float):
            return self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y,
                                          r.imu.gyro_z, self.target_heading, self.target_depth, surge, 0.0, roll, r.roll)
        self.last_action = a
        return self.step_with_driver(driver)

    # --- Judging ---
    def _err_deg(self) -> float:
        axis, angle = STAGES[min(self.stage, len(STAGES) - 1)]
        return (self.true_turned if axis == "yaw" else self.true_rolled) - angle

    def _physics_outcome(self) -> Optional[str]:
        s, dt = self.core.state, self.core.config.physicsDt
        self.true_turned += angle_diff(s.heading, self._last_true)
        self.true_rolled += angle_diff(s.roll, self._last_roll)
        self._last_true, self._last_roll = s.heading, s.roll
        if self.stage >= len(STAGES):
            return None
        err = self._err_deg()
        if abs(err) > 180.0:
            return "lost"
        axis, _ = STAGES[self.stage]
        if axis == "roll" and self.stage < len(STAGES) - 1:
            if abs(err) < ROLL_PASS_DEG:
                self.stage, self.held = self.stage + 1, 0.0
            return None
        rate = s.angular_velocity_z if axis == "yaw" else s.angular_velocity_x
        still = STILL_DPS if axis == "yaw" else 2 * STILL_DPS
        hold = HOLD_S if axis == "yaw" else LEVEL_HOLD_S
        if abs(err) < STOP_DEG and abs(math.degrees(rate)) < still:
            self.held += dt
            if self.held >= hold:
                self.stage, self.held = self.stage + 1, 0.0
        else:
            self.held = 0.0
        return None

    def done_label(self) -> bool:
        return self.stage >= len(STAGES)

    def _complete(self) -> bool:
        return self.done_label()

    def _terminal_reward(self, outcome: str) -> float:
        return 5.0 if outcome == "success" else -5.0

    def _potential(self) -> float:
        s = self.core.state
        drift = math.hypot(s.x - self.start_xy[0], s.y - self.start_xy[1])
        if self.stage >= len(STAGES):
            stages_part, off_axis = 3.0 * len(STAGES), 0.0
        else:
            axis, _ = STAGES[self.stage]
            hold = HOLD_S if axis == "yaw" else LEVEL_HOLD_S
            stages_part = 3.0 * self.stage - abs(self._err_deg()) / 90.0 + self.held / hold
            off_axis = abs(s.roll) / 90.0 if axis == "yaw" else abs(self.true_turned) / 90.0
        return stages_part - off_axis - 1.0 * abs(s.z - TARGET_DEPTH_M) - 0.5 * drift

    def _observe(self) -> np.ndarray:
        r = self.readings
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_z, r.roll, r.imu.gyro_x,
                                  np.asarray(self.last_action[:4]), self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _episode_info(self) -> dict:
        return {"stops": self.stage}

    def scripted_mission(self) -> List:
        """No scripted dance (it's learned only); the baseline just holds station, for train.py's comparison."""
        from ai.tasks.stabilize_task import StabilizeTask
        return [StabilizeTask(duration=self.max_seconds, target_depth=TARGET_DEPTH_M)]
