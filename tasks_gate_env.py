#!/usr/bin/env python3
"""
Gymnasium environment for the tasks 1-3 gate skill: go through the commanded
passage of the divided gate (left or right of the divider, relative to the
sub) and carry on 1.5 m beyond it. Starts face anywhere, so finding the gate
is part of the skill. The same skill makes the return pass: some episodes
start past the gate, facing back.

Reward (agreed with the user; the reasoning for each value is in the design
document's reward tables):
  progress   change in a score: minus the distance to the commanded passage's
             center (a point on the gate line, 0.76 m from the divider), plus
             1.0 x depth error from 1.4 m, plus 1.0 x sideways distance from
             the passage's center line, plus 1.0 per radian of heading away from
             straight through, phased in over the last 3 m. After the pass: its
             value then, +20, plus distance beyond the gate up to 1.5 m.
  success    +5, 1.5 m beyond the gate after passing through the right passage
  failures   -5 each: collision, wrong passage, missed (crossing the gate line
             outside the opening, or back again after passing), surfaced, timeout
  time       -0.01 per step;  energy  -0.05 per second at full thrust
The facing term was added after the first training runs went through crabwise,
45-90 deg off, with the sideways thrust: with one forward camera the sub then
can't see the divider it passes, and its square footprint turned 45 deg is 0.65
m across, in a passage with ~0.5 m to spare each side. It hit the divider.
No per-step cost on distance or angle during the approach: on the qualification
orbit one made crashing into the margin the cheapest way to end an episode.

Actions: turn, depth, surge (3), no sway. With sway the first policies slid
through crabwise, 45-90 deg off, and hit the divider; a facing cost didn't stop
it. Without sway the sub moves the way it faces, as the qualification gate
skill did. (Sliding through would be legal in competition.) Depth targets are limited to 1.25-1.65
m: shallower risks the top bar (down to 1.1 m), deeper leaves the camera too
close to the pole feet to range them.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import SubTaskEnv
from tasks_observation import TasksObservationBuilder

CLEAR_M = 1.5
PASSAGE_HALF_M = 1.524 / 2     # Passage center from the divider
TARGET_DEPTH_M = 1.4


@dataclass
class TasksGateStarts:
    return_share: float = 0.4             # Episodes that start past the gate, facing back
    forward_distance_m: tuple = (3.0, 10.0)
    return_distance_m: tuple = (2.0, 7.0)  # Short of slalom set 1 (8 m past the gate)
    lateral_m: tuple = (-2.0, 2.0)
    course_heading_share: float = 0.5     # Of forward starts: RoboSim's start headings (0, 90, 270 deg)
    any_heading_share: float = 0.3        # Of return starts: any heading; the rest within 30 deg of facing back
    depth_m: tuple = (0.5, 1.5)
    handoff_share: float = 0.0            # Of return starts: as the slalom hands over in the mission, just past
                                          # set 1 (~6 m out), near the lane's line, facing back within 20 deg and
                                          # already moving (the others start at rest)
    handoff_distance_m: tuple = (5.5, 7.0)
    handoff_lateral_m: tuple = (-0.4, 0.4)   # From the commanded passage's center line
    handoff_speed_mps: tuple = (0.2, 0.5)


@dataclass
class TasksGateReward:
    progress: float = 1.0
    depth_weight: float = 1.0
    center_line_weight: float = 1.0
    facing_weight: float = 1.0      # Per radian off straight through...
    facing_range_m: float = 3.0     # ...phased in from this far out to the gate line
    gate_bonus: float = 20.0
    clear_bonus: float = 5.0
    failure_penalty: float = 5.0
    time_penalty: float = 0.01
    energy_penalty: float = 0.05


class TasksGateEnv(SubTaskEnv):
    COURSE = "tasks"
    OUTCOMES = ("is_success", "collision", "wrong_passage", "missed", "surfaced", "timeout")
    ACTION_DIM = 3
    MIN_TARGET_DEPTH_M = 1.25
    MAX_TARGET_DEPTH_M = 1.65

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 90.0,
                 starts: Optional[TasksGateStarts] = None, reward: Optional[TasksGateReward] = None,
                 randomize: bool = False, side: Optional[int] = None, direction: Optional[int] = None):
        """side: +1 left / -1 right (relative to the sub) to fix it; None draws one per episode.
        direction: +1 forward / -1 return to fix it; None draws per starts.return_share."""
        self.starts = starts or TasksGateStarts()
        self.reward_cfg = rc = reward or TasksGateReward()
        self.fixed_side, self.fixed_direction = side, direction
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=rc.progress,
                         timeout_penalty=rc.failure_penalty, randomize=randomize)
        self.builder = TasksObservationBuilder(self.core.camera, "gate")
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    # --- Episode ---
    def _place_sub(self, rng: random.Random):
        s, st, info = self.core.state, self.starts, self.core.course.info
        gx, cy = info["gate_x"], info["gate_center_y"]
        self.direction = self.fixed_direction or (-1 if rng.random() < st.return_share else 1)
        self.side = self.fixed_side or rng.choice((1, -1))
        if self.direction > 0:
            s.x = gx - rng.uniform(*st.forward_distance_m)
            s.heading = (rng.choice((0.0, 90.0, 270.0)) if rng.random() < st.course_heading_share
                         else rng.uniform(-180.0, 180.0)) % 360
        else:
            s.x = gx + rng.uniform(*st.return_distance_m)
            s.heading = (rng.uniform(-180.0, 180.0) if rng.random() < st.any_heading_share
                         else 180.0 + rng.uniform(-30.0, 30.0)) % 360
        s.y = cy + rng.uniform(*st.lateral_m)
        s.z = rng.uniform(*st.depth_m)
        # The commanded passage on the course: relative left is the course's left going forward,
        # its right coming back
        self.lane = "left" if (self.side > 0) == (self.direction > 0) else "right"
        self.lane_y = cy + (PASSAGE_HALF_M if self.lane == "left" else -PASSAGE_HALF_M)
        if self.direction < 0 and rng.random() < st.handoff_share:
            s.x = gx + rng.uniform(*st.handoff_distance_m)
            s.y = self.lane_y + rng.uniform(*st.handoff_lateral_m)
            s.heading = (180.0 + rng.uniform(-20.0, 20.0)) % 360
            s.z = rng.uniform(1.0, 1.45)   # The slalom's depth limits
            v, h = rng.uniform(*st.handoff_speed_mps), math.radians(s.heading)
            s.velocity_x, s.velocity_y = v * math.cos(h), v * math.sin(h)

    def _on_reset(self):
        self.passed, self._at_pass, self._passes_seen = False, None, 0
        self.builder.reset()

    def _past_gate_m(self) -> float:
        return (self.core.state.x - self.core.course.info["gate_x"]) * self.direction

    def _physics_outcome(self) -> Optional[str]:
        new = self.core.passes[self._passes_seen:]
        self._passes_seen = len(self.core.passes)
        for p in new:
            if p.line != "gate":
                continue            # The slalom lines don't matter to this skill
            if self.passed or p.direction != self.direction or p.lane is None:
                return "missed"
            if p.lane != self.lane:
                return "wrong_passage"
            self.passed = True
        return None

    def _complete(self) -> bool:
        return self.passed and self._past_gate_m() >= CLEAR_M

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return rc.clear_bonus if outcome == "success" else -rc.failure_penalty

    def _approach_potential(self) -> float:
        s, rc, gx = self.core.state, self.reward_cfg, self.core.course.info["gate_x"]
        dist = math.hypot(gx - s.x, self.lane_y - s.y)
        through = 0.0 if self.direction > 0 else 180.0
        off = abs(math.radians((s.heading - through + 180.0) % 360.0 - 180.0))
        phase = min(max((rc.facing_range_m - dist) / rc.facing_range_m, 0.0), 1.0)
        return -(dist + rc.depth_weight * abs(s.z - TARGET_DEPTH_M)
                 + rc.center_line_weight * abs(s.y - self.lane_y) + rc.facing_weight * phase * off)

    def _potential(self) -> float:
        if self.passed:
            if self._at_pass is None:
                self._at_pass = self._approach_potential()
            return self._at_pass + self.reward_cfg.gate_bonus + min(max(self._past_gate_m(), 0.0), CLEAR_M)
        return self._approach_potential()

    def _observe(self) -> np.ndarray:
        r = self.readings
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                  self.detections, self.side, self.last_action, self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def scripted_mission(self) -> List:
        from ai.tasks.course_tasks import CourseGateTask
        return [CourseGateTask(target_depth=TARGET_DEPTH_M, side="left" if self.side > 0 else "right")]
