#!/usr/bin/env python3
"""
Gymnasium environment for the tasks 1-3 turnaround skill: after the forward
slalom, turn around to face the last set again, from far enough back to see
the commanded lane, so the slalom skill can take it from there. The runner says
"turn around" (what); this skill learns how (turning in place, backing off).

It was split from the slalom skill: started 1.5-3.5 m past set 3 facing away,
the slalom skill rarely learned to turn around (8-14%). Seeing a lane needs its
red pole and a white pole, 1.52 m apart, both in a 70 deg view, so from closer
than ~2.2 m no lane is visible even facing the set, and the slalom had to turn,
back off and line up mostly blind.

Starts: 1-3.5 m past set 3, facing on (away from it) within 30 deg, with the
set remembered as the forward pass left it (TasksSlalomEnv._remember_set; in
the mission the slalom's memory carries over).

Done (ground truth): about 3 m from the set's line (within 0.75 m of a point
3 m back, in front of the commanded lane's center), facing straight back
within 15 deg, with the red pole and the commanded side's white pole both
inside the camera's view.

Reward:
  progress   change in a score: minus the distance to that point, minus 1.0 per
             radian off facing straight back, minus 1.0 x depth error from 1.25 m
  success    +20 (the job is the big payoff)
  failures   -5 each: collision, crossing the set's line (missed), surfaced, timeout
  time       -0.01 per step;  energy  -0.05 per second at full thrust
Actions: turn, depth, surge (3), no sway, as the other tasks skills.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

from tasks_slalom_env import LANE_HALF_M, TARGET_DEPTH_M, TasksSlalomEnv

BACK_M = 3.0               # How far from the set's line to end up
DONE_RADIUS_M = 0.75
DONE_FACING_DEG = 15.0
HALF_FOV_DEG = 35.0


@dataclass
class TurnaroundStarts:
    distance_m: tuple = (1.0, 3.5)
    lateral_m: tuple = (-1.2, 1.2)
    heading_spread_deg: float = 30.0
    depth_m: tuple = (1.0, 1.4)


@dataclass
class TurnaroundReward:
    facing_weight: float = 1.0
    depth_weight: float = 1.0
    success_bonus: float = 20.0
    failure_penalty: float = 5.0
    time_penalty: float = 0.01
    energy_penalty: float = 0.05


class TasksTurnaroundEnv(TasksSlalomEnv):
    OUTCOMES = ("is_success", "collision", "missed", "surfaced", "timeout")

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 45.0,
                 starts: Optional[TurnaroundStarts] = None, reward: Optional[TurnaroundReward] = None,
                 randomize: bool = False, side: Optional[int] = None):
        self.turn_starts = starts or TurnaroundStarts()
        self.turn_reward = rw = reward or TurnaroundReward()
        super().__init__(detector_mode, control_period, max_seconds, randomize=randomize, side=side)
        self.time_penalty, self.energy_penalty = rw.time_penalty, rw.energy_penalty
        self.progress_weight, self.timeout_penalty = 1.0, rw.failure_penalty

    def _place_sub(self, rng: random.Random):
        s, st, info = self.core.state, self.turn_starts, self.core.course.info
        self.turnaround = True
        self.set_index, self.direction = 2, -1          # Coming back through set 3
        self.set_x, self.set_y = info["slalom_centers"][2]
        self.side = self.fixed_side or rng.choice((1, -1))
        s.x = self.set_x + rng.uniform(*st.distance_m)
        s.y = self.set_y + rng.uniform(*st.lateral_m)
        s.heading = rng.uniform(-st.heading_spread_deg, st.heading_spread_deg) % 360   # Facing on, away from it
        s.z = rng.uniform(*st.depth_m)
        self.lane = "right" if self.side > 0 else "left"   # Relative left, facing back, is the course's right
        self.lane_y = self.set_y + (LANE_HALF_M if self.lane == "left" else -LANE_HALF_M)
        self.white_y = self.set_y + (2 * LANE_HALF_M if self.lane == "left" else -2 * LANE_HALF_M)
        self.target = (self.set_x + BACK_M, self.lane_y)
        self.line = "slalom 3"

    def _physics_outcome(self) -> Optional[str]:
        new = self.core.passes[self._passes_seen:]
        self._passes_seen = len(self.core.passes)
        if any(p.line == self.line for p in new):
            return "missed"        # This skill only turns around; going through is the slalom skill's job
        return None

    def _facing_off_rad(self) -> float:
        return abs(math.radians((self.core.state.heading - 180.0 + 180.0) % 360.0 - 180.0))

    def _in_view(self, x: float, y: float) -> bool:
        s = self.core.state
        bearing = (math.degrees(math.atan2(y - s.y, x - s.x)) - s.heading + 180.0) % 360.0 - 180.0
        return abs(bearing) < HALF_FOV_DEG - 3.0

    def _complete(self) -> bool:
        s = self.core.state
        return (math.hypot(self.target[0] - s.x, self.target[1] - s.y) < DONE_RADIUS_M
                and math.degrees(self._facing_off_rad()) < DONE_FACING_DEG
                and self._in_view(self.set_x, self.set_y) and self._in_view(self.set_x, self.white_y))

    def _terminal_reward(self, outcome: str) -> float:
        rw = self.turn_reward
        return rw.success_bonus if outcome == "success" else -rw.failure_penalty

    def _potential(self) -> float:
        s, rw = self.core.state, self.turn_reward
        return -(math.hypot(self.target[0] - s.x, self.target[1] - s.y) + rw.facing_weight * self._facing_off_rad()
                 + rw.depth_weight * abs(s.z - TARGET_DEPTH_M))

    def scripted_mission(self) -> List:
        from ai.tasks.course_tasks import SlalomTask
        offset = self.readings.heading - self.core.state.heading
        return [SlalomTask(target_depth=TARGET_DEPTH_M, side="left" if self.side > 0 else "right",
                           axis=(180.0 + offset) % 360)]
