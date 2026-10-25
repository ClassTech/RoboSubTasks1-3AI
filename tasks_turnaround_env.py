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

Starts: most begin with a lead-in: the slalom skill flies through set 3 from
1.5-3 m before it, with a fresh observation, and hands over when its done
detector fires, as in the mission. The turnaround then starts with the memory,
speed and place the slalom really leaves. The first version filled the memory
from ground truth instead (the set's current range, seen 2 s ago); the mission
leaves the range when last seen (~1 m: memory isn't corrected for moving) and
seen 5-10 s ago, and there the turnaround's detector fired at once, still
facing away. The rest (and any failed lead-in): 1-3.5 m past set 3, facing on
(away from it) within 30 deg, with the set remembered that way
(TasksSlalomEnv._remember_set).

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

import numpy as np

from tasks_slalom_env import LANE_HALF_M, TARGET_DEPTH_M, TasksSlalomEnv

LEAD_IN_POLICY = "runs/tslalom6_ft_s1/best_model.zip"   # With its done detector beside it
LEAD_IN_MAX_S = 40.0
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
    lead_in_share: float = 0.75


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

    # --- Lead-in: the slalom skill through set 3, as in the mission ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        if self.np_random.random() < self.turn_starts.lead_in_share and self._lead_in():
            obs = self._start_after_lead_in()
        return obs, info

    def _lead_in_skill(self):
        if not hasattr(self, "_lead_in_parts"):
            from done_detector import DoneDetector, detector_path
            from gate_controllers import load_policy
            model, norm = load_policy(LEAD_IN_POLICY)
            self._lead_in_parts = (model, norm, DoneDetector(detector_path(LEAD_IN_POLICY)))
        return self._lead_in_parts

    def _drive(self, action) -> bool:
        """One control period toward the action's setpoints, without this skill's scoring. False on a collision."""
        from sub_env import DEPTH_MAX, DEPTH_MIN, HEADING_CHANGE_DEG
        a = np.clip(action, -1.0, 1.0)
        heading = (self.readings.heading + HEADING_CHANGE_DEG * float(a[0])) % 360
        depth = min(max(DEPTH_MIN + (float(a[1]) + 1) / 2 * (DEPTH_MAX - DEPTH_MIN), self.MIN_TARGET_DEPTH_M),
                    self.MAX_TARGET_DEPTH_M)
        dt, hits = self.core.config.physicsDt, self.core.collision_count
        for _ in range(max(1, round(self.control_period / dt))):
            r = self.readings
            cmd = self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                         heading, depth, float(a[2]), 0.0)
            self.core.advance(dt, cmd)
            self.readings, self.detections = self.core.read_sensors(dt), self.core.detect(dt)
        return self.core.collision_count == hits

    def _lead_in(self) -> bool:
        """Flies the slalom skill through set 3 (going out, in this episode's lane) until its detector fires.
        False if it doesn't get through cleanly; the episode then keeps its ordinary start."""
        model, norm, detector = self._lead_in_skill()
        rng, s = self.np_random, self.core.state
        s.x = self.set_x - rng.uniform(1.5, 3.0)
        s.y = self.lane_y + rng.uniform(-0.5, 0.5)
        s.heading, s.z = rng.uniform(-10.0, 10.0) % 360, rng.uniform(1.0, 1.4)
        s.velocity_x = s.velocity_y = s.velocity_z = s.angular_velocity_z = 0.0
        self.readings, self.detections = self.core.read_sensors(0.0), self.core.detect(0.0)
        passes = len(self.core.passes)
        self.builder.reset()
        detector.reset()
        action, out_side = np.zeros(3, dtype=np.float32), -self.side   # Going out, the lane is on the other side
        for _ in range(int(LEAD_IN_MAX_S / self.control_period)):
            r = self.readings
            obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                      self.detections, out_side, action, self.control_period)
            if detector.update(obs, r.heading):
                new = self.core.passes[passes:]
                return [(p.line, p.direction, p.lane) for p in new] == [(self.line, 1, self.lane)]
            action = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1.0, 1.0)
            if not self._drive(action):
                return False
        return False

    def _start_after_lead_in(self) -> np.ndarray:
        """The turnaround takes over as in the mission: the slalom's tracks kept, its frames and start heading not."""
        from collections import deque
        b = self.builder
        b.start_heading, b.frames = None, deque(maxlen=b.stack)
        self.target_heading, self.target_depth = self.readings.heading, self.readings.depth
        self.time, self.energy = 0.0, 0.0
        self._passes_seen, self._collisions_seen = len(self.core.passes), self.core.collision_count
        self._submerged = True
        self.potential = self._potential()
        self.last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        return self._observe()

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
