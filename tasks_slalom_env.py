#!/usr/bin/env python3
"""
Gymnasium environment for the tasks 1-3 slalom skill: go through one slalom set
in the commanded lane (left or right of its red pole, relative to the sub) and
carry on 1.5 m beyond it. The mission runs it three times each way; the
starts include facing away, so turning around after the third set is part of
the skill, not the runner's.

Reward (the gate skill's structure, agreed with the user):
  progress   change in a score: minus the distance to the commanded lane's
             center (a point on the set's line, 0.76 m from its red pole), plus
             1.0 x depth error from 1.25 m, plus 1.0 x sideways distance from
             the lane's center line, plus 1.0 per radian off straight through,
             phased in over the last 2 m. After the pass: its value then, +20,
             plus distance beyond the set up to 1.5 m.
  success    +5, 1.5 m beyond the set after passing through the right lane
  failures   -5 each: collision, wrong lane, missed (crossing the set's line
             outside its lanes, including over or under the floating poles, or
             back again after passing), backwards (crossing it facing more than
             90 deg away from the way through), surfaced, timeout
The backwards rule closed a loophole: on the turnaround starts a policy learned
to reverse blind through set 3 instead of turning to see it, picking a lane by
luck. (Backing through would count in competition, but with one forward camera
it's always blind.)
  time       -0.01 per step;  energy  -0.05 per second at full thrust

Actions: turn, depth, surge (3), no sway: with sway the gate skill went
through crabwise. Depth targets are limited to 1.0-1.45 m: the poles float
with their tops 0.6-0.9 m deep and bottoms 1.5-1.8 m, and a lane only counts
inside their span.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import SubTaskEnv
from tasks_observation import TasksObservationBuilder

CLEAR_M = 1.5
LANE_HALF_M = 1.524 / 2
TARGET_DEPTH_M = 1.25


@dataclass
class TasksSlalomStarts:
    distance_m: tuple = (1.5, 3.5)        # Before the set, along the direction of travel. Less than the 4 m
                                          # between sets, so no other set is ever in the way
    lateral_m: tuple = (-1.5, 1.5)        # From the set's red pole
    heading_spread_deg: float = 30.0      # Facing the set within this...
    turnaround_share: float = 0.0         # ...except these: past set 3, facing on (away from it), to come back
                                          # through it, with the set remembered (_remember_set). Off: at 0.3-0.5
                                          # the skill turned around only 8-14% of the time and got worse at the
                                          # rest (88% -> 73%), so turning around is its own skill now
                                          # (tasks_turnaround_env.py)
    depth_m: tuple = (1.0, 1.4)


@dataclass
class TasksSlalomReward:
    progress: float = 1.0
    depth_weight: float = 1.0
    center_line_weight: float = 2.0   # Was 1.0: policies kept shaving the red pole, the lane's inner edge
    facing_weight: float = 1.0
    facing_range_m: float = 2.0
    lane_bonus: float = 20.0
    clear_bonus: float = 5.0
    failure_penalty: float = 5.0
    time_penalty: float = 0.01
    energy_penalty: float = 0.05


class TasksSlalomEnv(SubTaskEnv):
    COURSE = "tasks"
    OUTCOMES = ("is_success", "collision", "wrong_lane", "missed", "backwards", "surfaced", "timeout")
    ACTION_DIM = 3
    MIN_TARGET_DEPTH_M = 1.0
    MAX_TARGET_DEPTH_M = 1.45

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 60.0,
                 starts: Optional[TasksSlalomStarts] = None, reward: Optional[TasksSlalomReward] = None,
                 randomize: bool = False, side: Optional[int] = None):
        self.starts = starts or TasksSlalomStarts()
        self.reward_cfg = rc = reward or TasksSlalomReward()
        self.fixed_side = side
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=rc.progress,
                         timeout_penalty=rc.failure_penalty, randomize=randomize)
        self.builder = TasksObservationBuilder(self.core.camera, "slalom")
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    # --- Episode ---
    def _place_sub(self, rng: random.Random):
        s, st, info = self.core.state, self.starts, self.core.course.info
        turnaround = self.turnaround = rng.random() < st.turnaround_share
        self.set_index = 2 if turnaround else rng.randrange(3)
        self.set_x, self.set_y = info["slalom_centers"][self.set_index]
        self.direction = -1 if turnaround else rng.choice((1, -1))
        self.side = self.fixed_side or rng.choice((1, -1))
        s.x = self.set_x - self.direction * rng.uniform(*st.distance_m)
        s.y = self.set_y + rng.uniform(*st.lateral_m)
        facing = 0.0 if self.direction > 0 else 180.0
        if turnaround:
            facing += 180.0   # Still facing the way the forward slalom went
        s.heading = (facing + rng.uniform(-st.heading_spread_deg, st.heading_spread_deg)) % 360
        s.z = rng.uniform(*st.depth_m)
        self.lane = "left" if (self.side > 0) == (self.direction > 0) else "right"
        self.lane_y = self.set_y + (LANE_HALF_M if self.lane == "left" else -LANE_HALF_M)
        self.line = f"slalom {self.set_index + 1}"

    def _on_reset(self):
        self.passed, self._at_pass, self._passes_seen = False, None, 0
        self.builder.reset()
        if self.turnaround:
            self._remember_set()

    def _remember_set(self, seen_ago_s: float = 2.0):
        """
        Turnaround starts: the sub has just come through this set going forward, so it saw it seconds ago.
        Fill the observation's memory as the forward pass would have left it (labels as seen facing forward:
        'left' is the course's left). In the mission the slalom keeps its memory from forward to back.
        Without this, the sub started blind with the set behind it, and learned to reverse through blind.
        """
        s, r, b = self.core.state, self.readings, self.builder
        offset = r.heading - s.heading            # Measured minus true heading (the magnetometer's error)

        def remember(track, x, y):
            bearing = (math.degrees(math.atan2(y - s.y, x - s.x)) - s.heading + 180.0) % 360.0 - 180.0
            track.seen_ever, track.found = True, False
            track.abs_bearing = s.heading + offset + bearing
            track.range_m, track.confidence, track.unseen = math.hypot(x - s.x, y - s.y), 0.8, seen_ago_s
            return bearing

        red = remember(b.red, self.set_x, self.set_y)
        for side, sign in (("left", 1), ("right", -1)):
            white = remember(b.whites[side], self.set_x, self.set_y + sign * 2 * LANE_HALF_M)
            remember(b.lanes[side], self.set_x, self.set_y + sign * LANE_HALF_M)
            b.lanes[side].width_deg = abs(red - white)

    def _past_set_m(self) -> float:
        return (self.core.state.x - self.set_x) * self.direction

    def _physics_outcome(self) -> Optional[str]:
        new = self.core.passes[self._passes_seen:]
        self._passes_seen = len(self.core.passes)
        for p in new:
            if p.line != self.line:
                continue            # Other sets and the gate don't matter to this episode
            through = 0.0 if p.direction > 0 else 180.0
            if abs((self.core.state.heading - through + 180.0) % 360.0 - 180.0) > 90.0:
                return "backwards"
            if self.passed or p.direction != self.direction or p.lane is None:
                return "missed"
            if p.lane != self.lane:
                return "wrong_lane"
            self.passed = True
        return None

    def _complete(self) -> bool:
        return self.passed and self._past_set_m() >= CLEAR_M

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return rc.clear_bonus if outcome == "success" else -rc.failure_penalty

    def _approach_potential(self) -> float:
        s, rc = self.core.state, self.reward_cfg
        dist = math.hypot(self.set_x - s.x, self.lane_y - s.y)
        through = 0.0 if self.direction > 0 else 180.0
        off = abs(math.radians((s.heading - through + 180.0) % 360.0 - 180.0))
        phase = min(max((rc.facing_range_m - dist) / rc.facing_range_m, 0.0), 1.0)
        return -(dist + rc.depth_weight * abs(s.z - TARGET_DEPTH_M)
                 + rc.center_line_weight * abs(s.y - self.lane_y) + rc.facing_weight * phase * off)

    def _potential(self) -> float:
        if self.passed:
            if self._at_pass is None:
                self._at_pass = self._approach_potential()
            return self._at_pass + self.reward_cfg.lane_bonus + min(max(self._past_set_m(), 0.0), CLEAR_M)
        return self._approach_potential()

    def _observe(self) -> np.ndarray:
        r = self.readings
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                  self.detections, self.side, self.last_action, self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def scripted_mission(self) -> List:
        from ai.tasks.course_tasks import SlalomTask
        # The course heading in the measured heading's frame (the magnetometer is off by up to 15 deg per run)
        offset = self.readings.heading - self.core.state.heading
        axis = ((0.0 if self.direction > 0 else 180.0) + offset) % 360
        return [SlalomTask(target_depth=TARGET_DEPTH_M, side="left" if self.side > 0 else "right", axis=axis)]
