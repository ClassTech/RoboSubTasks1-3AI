#!/usr/bin/env python3
"""
RoboSub tasks 1-3 on the tasks course (course.py): through one side of the
divided gate, through the three slalom sets on the same side, then back the
same way, through the same lanes.

build_tasks_mission() is the scripted mission, from ClassTech/RoboSim's main.py
(the victory dance left out for now): gate, stabilize, slalom, stabilize,
slalom reversed, gate, hover 10 s, surface.

TasksMissionEnv judges a whole run on ground truth, from the sim's passes:
success needs, in order, the gate forward in some lane, slalom sets 1, 2 and 3
forward in that same lane (the same side as the gate), then sets 3, 2 and 1
back and the gate back, all in that same lane. Failures: a collision; crossing
a gate or set line outside its lanes (missed); a lane other than the gate's
(wrong_lane); crossing lines out of order, e.g. skipping a set (wrong_course);
surfacing; running out of time.

TasksMissionRunner (below) is the learned mission: the gate, slalom and
turnaround skills in a plain sequence, each until its done detector fires.
Score it with tools/eval_gate.py tasks_mission --task tasks_mission.
"""
import random
from collections import deque
from typing import List, Optional

import numpy as np

from sub_env import DEPTH_MAX, DEPTH_MIN, SubTaskEnv

MISSION_DEPTH = 1.3   # Below the deepest gate top bar (1.1 m) and within the floating slalom poles (down to 1.5 m)
ROUTE = [("gate", 1), ("slalom 1", 1), ("slalom 2", 1), ("slalom 3", 1),
         ("slalom 3", -1), ("slalom 2", -1), ("slalom 1", -1), ("gate", -1)]


def build_tasks_mission(depth: float = MISSION_DEPTH) -> List:
    from ai.tasks.course_tasks import CourseGateTask, SlalomTask, SurfaceTask, TimedHoverTask
    from ai.tasks.shutdown_task import ShutdownTask
    from ai.tasks.stabilize_task import StabilizeTask
    return [
        CourseGateTask(target_depth=depth),
        StabilizeTask(duration=3.0, target_depth=depth),
        SlalomTask(target_depth=depth),
        StabilizeTask(duration=3.0, target_depth=depth),
        SlalomTask(target_depth=depth, reversed=True),
        CourseGateTask(target_depth=depth),
        TimedHoverTask(duration=10.0, target_depth=depth),
        SurfaceTask(),
        ShutdownTask(),
    ]


class TasksMissionEnv(SubTaskEnv):
    COURSE = "tasks"
    OUTCOMES = ("is_success", "collision", "missed", "wrong_lane", "wrong_course", "surfaced", "timeout")
    EXTRA_INFO = ("route_done",)   # Fraction of the eight gate and set passes completed
    ACTION_DIM = 4

    def __init__(self, detector_mode: str = "noisy", max_seconds: float = 600.0, randomize: bool = False):
        super().__init__(detector_mode, max_seconds=max_seconds, timeout_penalty=0.0, randomize=randomize)

    def _place_sub(self, rng: random.Random):
        pass   # SubSim puts the sub at the course's random start

    def _on_reset(self):
        self.done, self.lane, self._passes_seen = 0, None, 0

    def _physics_outcome(self) -> Optional[str]:
        new = self.core.passes[self._passes_seen:]
        self._passes_seen = len(self.core.passes)
        for p in new:
            if p.lane is None:
                return "missed"
            if (p.line, p.direction) != ROUTE[self.done]:
                return "wrong_course"
            self.lane = self.lane or p.lane
            if p.lane != self.lane:
                return "wrong_lane"
            self.done += 1
            if self.done == len(ROUTE):
                return "success"
        return None

    def _potential(self) -> float:
        return 0.0   # Only for scoring, not training

    def _terminal_reward(self, outcome: str) -> float:
        return 0.0

    def _episode_info(self) -> dict:
        return {"route_done": self.done / len(ROUTE)}

    def scripted_mission(self) -> List:
        return build_tasks_mission()


GATE_POLICY = "runs/tgate3_s0/best_model.zip"
SLALOM_POLICY = "runs/tslalom13_whole_s13/best_model.zip"
TURNAROUND_POLICY = "runs/tturn4_ft_s5/best_model.zip"
OUT, BACK = 1, -1      # The side relative to the side chosen at the gate: the same, or (coming back) the other
PLAN = [("gate", OUT), ("slalom", OUT), ("turnaround", BACK), ("slalom", BACK), ("gate", BACK)]
DEPTH_LIMITS = {"gate": (1.25, 1.65), "slalom": (1.0, 1.45), "turnaround": (1.0, 1.45)}   # As each trained
HOLD_DEPTH_M = 1.3


class TasksMissionRunner:
    """
    The learned tasks 1-3 mission. The runner says what to do, never how: a plain
    list of (skill, side) steps, each flown by its learned policy until that
    skill's done detector (done_detector.py) says it's finished. Then it holds
    depth and heading (the scripted mission hovers 10 s, then surfaces).

    The side is chosen as the scripted mission chooses it: the passage nearer the
    image center when the gate's two passages are first both in view. Until then
    the gate skill is told "left" (it's still searching). Coming back, the same
    physical lanes are on the sub's other side.

    Observations: each step starts its skill with a fresh observation, as in
    training, except the turnaround, which keeps the slalom's memory of set 3
    (in training its memory is filled as the forward slalom would have left it).
    """
    name = "tasks_mission"

    def __init__(self, gate_path: str = GATE_POLICY, slalom_path: str = SLALOM_POLICY,
                 turnaround_path: str = TURNAROUND_POLICY):
        from done_detector import DoneDetector, detector_path
        from gate_controllers import load_policy
        self.skills = {}
        for skill, path in (("gate", gate_path), ("slalom", slalom_path), ("turnaround", turnaround_path)):
            model, norm = load_policy(path)
            self.skills[skill] = (model, norm, DoneDetector(detector_path(path)))
        self.last_action = None

    def reset(self, env):
        from tasks_observation import TasksObservationBuilder
        cam = env.core.camera
        self.builders = {"gate": TasksObservationBuilder(cam, "gate"), "slalom": TasksObservationBuilder(cam, "slalom")}
        self.side = None                 # +1 left / -1 right, relative to the sub going out
        self.step_index = -1
        self.hold_heading = None
        self.switch_times = []
        self.last_action = None
        self._next(env)

    def state_name(self) -> str:
        if self.step_index >= len(PLAN):
            return "DONE: holding"
        skill, way = PLAN[self.step_index]
        label = {"gate": "GATE", "slalom": "SLALOM", "turnaround": "TURN AROUND"}[skill]
        side = "" if self.side is None else f" ({'left' if self._side() > 0 else 'right'})"
        return f"{label} {'out' if way == OUT else 'back'}{side}"

    def _side(self) -> int:
        return (self.side or 1) * PLAN[self.step_index][1]

    def _builder(self, skill):
        return self.builders["gate" if skill == "gate" else "slalom"]

    def _next(self, env):
        self.step_index += 1
        self.switch_times.append(env.time)
        self.skill_action = np.zeros(3, dtype=np.float32)
        if self.step_index >= len(PLAN):
            self.hold_heading = env.readings.heading
            return
        skill = PLAN[self.step_index][0]
        b = self._builder(skill)
        if skill == "turnaround":
            b.start_heading, b.frames = None, deque(maxlen=b.stack)   # Keeps its tracks of set 3
        else:
            b.reset()
        self.skills[skill][2].reset()

    def _choose_side(self, b, heading: float):
        """The scripted rule: the passage nearer the image center, once both passages are in view."""
        p = b.passages
        if self.side is None and p["left"].found and p["right"].found:
            self.side = 1 if abs(p["left"].bearing(heading)) <= abs(p["right"].bearing(heading)) else -1

    def _act(self, env):
        skill = PLAN[self.step_index][0]
        model, norm, detector = self.skills[skill]
        r, d, b = env.readings, env.detections, self._builder(skill)
        obs = b.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z, d, self._side(),
                       np.clip(self.skill_action, -1.0, 1.0), env.control_period)
        self.last_obs = obs   # The current skill's observation (for viewing and debugging)
        if skill == "gate":
            self._choose_side(b, r.heading)
        a = np.asarray(model.predict(norm.normalize_obs(obs), deterministic=True)[0], dtype=np.float32)
        return a, detector.update(obs, r.heading)

    @staticmethod
    def _depth_action(depth_m: float) -> float:
        return (depth_m - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1

    def step(self, env, obs):
        if self.step_index < len(PLAN):
            self.skill_action, done = self._act(env)
            if done:
                self._next(env)
                if self.step_index < len(PLAN):
                    self.skill_action, _ = self._act(env)   # The next skill acts on this step already
        if self.step_index < len(PLAN):
            lo, hi = DEPTH_LIMITS[PLAN[self.step_index][0]]
            action = np.append(np.clip(self.skill_action, -1.0, 1.0), 0.0)   # No sway
            action[1] = float(np.clip(action[1], self._depth_action(lo), self._depth_action(hi)))
        else:
            turn = float(np.clip(((self.hold_heading - env.readings.heading + 180) % 360 - 180) / 45.0, -1, 1))
            action = np.array([turn, self._depth_action(HOLD_DEPTH_M), 0.0, 0.0], dtype=np.float32)
        self.last_action = action
        return env.step(action)
