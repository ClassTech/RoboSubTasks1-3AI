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
surfacing; running out of time. The victory dance (tasks_dance_env.py) is judged
too, from when the runner says it starts, as dance_stops (0-16): its points are
extra, so a dance cut short or lost doesn't fail the run.

TasksMissionRunner (below) is the learned mission: the gate, dance, slalom and
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
    EXTRA_INFO = ("route_done", "dance_stops")   # Fraction of the eight gate and set passes completed; dance stops
    ACTION_DIM = 4

    def __init__(self, detector_mode: str = "noisy", max_seconds: float = 600.0, randomize: bool = False):
        super().__init__(detector_mode, max_seconds=max_seconds, timeout_penalty=0.0, randomize=randomize)

    def _place_sub(self, rng: random.Random):
        pass   # SubSim puts the sub at the course's random start

    def _on_reset(self):
        self.done, self.lane, self._passes_seen = 0, None, 0
        self.dance_judge, self.dance_stops = None, 0

    def start_dance(self):
        """The runner starts the dance: the judges count its stops from the heading the sub has now."""
        from tasks_dance_env import DanceJudge
        self.dance_judge = DanceJudge(self.core.state)

    def end_dance(self):
        self.dance_judge = None

    def _physics_outcome(self) -> Optional[str]:
        if self.dance_judge:
            self.dance_judge.update(self.core.state, self.core.config.physicsDt)
            self.dance_stops = self.dance_judge.stage
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
        return {"route_done": self.done / len(ROUTE), "dance_stops": self.dance_stops}

    def scripted_mission(self) -> List:
        return build_tasks_mission()


GATE_POLICY = "runs/tgate5_785_s25/best_model.zip"
SLALOM_POLICY = "runs/tslalom18_785_s23/best_model.zip"
TURNAROUND_POLICY = "runs/tturn8_785_s22/best_model.zip"
DANCE_POLICY = "runs/tdance_bc8/best_model.zip"
DANCE_GIVE_UP_S = 120.0   # The dance's time limit in training
SETTLE_DEPTH_M, SETTLE_DPS, SETTLE_GIVE_UP_S = 0.1, 0.1, 10.0   # After the dance: back to HOLD_DEPTH_M before the slalom
OUT, BACK = 1, -1      # The side relative to the side chosen at the gate: the same, or (coming back) the other
PLAN = [("gate", OUT), ("dance", OUT), ("slalom", OUT), ("turnaround", BACK), ("slalom", BACK), ("gate", BACK)]
DEPTH_LIMITS = {"gate": (1.25, 1.65), "slalom": (1.0, 1.45), "turnaround": (1.0, 1.45)}   # As each trained
HOLD_DEPTH_M = 1.3


class TasksMissionRunner:
    """
    The learned tasks 1-3 mission. The runner says what to do, never how: a plain
    list of (skill, side) steps, each flown by its learned policy until that
    skill's done detector (done_detector.py) says it's finished. Then it holds
    depth and heading (the scripted mission hovers 10 s, then surfaces).

    The dance has no done detector: its own observation says when it's back level
    after the full roll back (the last stop), and it gives up after DANCE_GIVE_UP_S.
    Then it rises back to HOLD_DEPTH_M on the heading the dance started on: the dance
    ends ~0.4 m deeper (it rolls deep), and the slalom started from there failed the
    way back 3x as often (277/300 against 286/300 without the dance).
    It flies through its own driver (tasks_dance_env.DanceDriver: roll, and the heading
    gains halved, as it trained).

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
                 turnaround_path: str = TURNAROUND_POLICY, dance_path: Optional[str] = DANCE_POLICY):
        """dance_path None leaves the dance out."""
        from done_detector import DoneDetector, detector_path
        from gate_controllers import load_policy
        self.skills = {}
        for skill, path in (("gate", gate_path), ("slalom", slalom_path), ("turnaround", turnaround_path)):
            model, norm = load_policy(path)
            self.skills[skill] = (model, norm, DoneDetector(detector_path(path)))
        self.plan = PLAN if dance_path else [step for step in PLAN if step[0] != "dance"]
        self.dance_policy = load_policy(dance_path) if dance_path else None
        self.last_action = None

    def reset(self, env):
        from tasks_observation import TasksObservationBuilder
        cam = env.core.camera
        from tasks_dance_env import DanceDriver, DanceObservation
        self.builders = {"gate": TasksObservationBuilder(cam, "gate"), "slalom": TasksObservationBuilder(cam, "slalom")}
        self.dance_obs, self.dancer = DanceObservation(), DanceDriver(env.controller, env.control_period)
        self.normal_gains = None
        self.side = None                 # +1 left / -1 right, relative to the sub going out
        self.step_index = -1
        self.hold_heading = None
        self.switch_times = []
        self.last_action = None
        self._next(env)

    def state_name(self) -> str:
        if self.step_index >= len(self.plan):
            return "DONE: holding"
        skill, way = self.plan[self.step_index]
        if skill == "dance":
            return "DANCE: rising" if self.settle_start is not None else f"DANCE ({self.dance_obs_stop()})"
        label = {"gate": "GATE", "slalom": "SLALOM", "turnaround": "TURN AROUND"}[skill]
        side = "" if self.side is None else f" ({'left' if self._side() > 0 else 'right'})"
        return f"{label} {'out' if way == OUT else 'back'}{side}"

    def _side(self) -> int:
        return (self.side or 1) * self.plan[self.step_index][1]

    def _builder(self, skill):
        return self.builders["gate" if skill == "gate" else "slalom"]

    def _skill(self) -> Optional[str]:
        return self.plan[self.step_index][0] if self.step_index < len(self.plan) else None

    def dance_obs_stop(self) -> str:
        o = self.dance_obs
        return f"turned {o.turned:.0f}, rolled {o.rolled:.0f}"

    def _next(self, env):
        if self._skill() == "dance":
            env.end_dance()
            env.controller.gains = self.normal_gains
        self.step_index += 1
        self.switch_times.append(env.time)
        self.skill_action = np.zeros(3, dtype=np.float32)
        if self.step_index >= len(self.plan):
            self.hold_heading = env.readings.heading
            return
        skill = self.plan[self.step_index][0]
        if skill == "dance":
            self._start_dance(env)
            return
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
        skill = self.plan[self.step_index][0]
        model, norm, detector = self.skills[skill]
        r, d, b = env.readings, env.detections, self._builder(skill)
        obs = b.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z, d, self._side(),
                       np.clip(self.skill_action, -1.0, 1.0), env.control_period)
        self.last_obs = obs   # The current skill's observation (for viewing and debugging)
        if skill == "gate":
            self._choose_side(b, r.heading)
        a = np.asarray(model.predict(norm.normalize_obs(obs), deterministic=True)[0], dtype=np.float32)
        return a, detector.update(obs, r.heading)

    def _start_dance(self, env):
        from dataclasses import replace
        from tasks_dance_env import HEADING_GAIN_SCALE
        env.start_dance()
        self.dance_obs.reset(stop_done=True)   # The start needs no hold
        self.dancer.reset()
        self.dance_start, self.dance_heading, self.settle_start = env.time, env.readings.heading, None
        self.normal_gains = env.controller.gains
        g = self.normal_gains
        env.controller.gains = replace(g, heading_p=g.heading_p * HEADING_GAIN_SCALE, yaw_d=g.yaw_d * HEADING_GAIN_SCALE)

    def _dance_done(self, env) -> bool:
        """Back level after the full roll back, held as the last stop's hold plus a margin; or out of time."""
        from tasks_dance_env import LEVEL_HOLD_S, ROLL_PASS_DEG
        o = self.dance_obs
        level_again = o.rolled_most > 360.0 - ROLL_PASS_DEG and o.level >= LEVEL_HOLD_S + 0.5
        return level_again or env.time - self.dance_start >= DANCE_GIVE_UP_S

    def _dance_step(self, env):
        model, norm = self.dance_policy
        r = env.readings
        obs = self.dance_obs.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_z, r.roll, r.imu.gyro_x,
                                    self.dancer.turn_target, env.control_period)
        self.last_obs = obs
        a = np.asarray(model.predict(norm.normalize_obs(np.clip(obs, -10.0, 10.0)), deterministic=True)[0],
                       dtype=np.float32)
        env.target_heading, env.target_depth, driver = self.dancer.act(a, r.heading, self.dance_obs.turned)
        self.last_action = np.clip(a, -1.0, 1.0)
        return env.step_with_driver(driver)

    @staticmethod
    def _depth_action(depth_m: float) -> float:
        return (depth_m - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1

    def _hold_action(self, env, heading: float) -> np.ndarray:
        turn = float(np.clip(((heading - env.readings.heading + 180) % 360 - 180) / 45.0, -1, 1))
        return np.array([turn, self._depth_action(HOLD_DEPTH_M), 0.0, 0.0], dtype=np.float32)

    def _settled(self, env) -> bool:
        r = env.readings
        back = abs(r.depth - HOLD_DEPTH_M) < SETTLE_DEPTH_M and abs(r.depth_rate) < SETTLE_DPS
        return back or env.time - self.settle_start >= SETTLE_GIVE_UP_S

    def step(self, env, obs):
        if self._skill() == "dance":
            if self.settle_start is None:
                if not self._dance_done(env):
                    return self._dance_step(env)
                self.settle_start = env.time
                env.controller.gains = self.normal_gains
            if not self._settled(env):
                self.last_action = self._hold_action(env, self.dance_heading)
                return env.step(self.last_action)
            self._next(env)
            self.skill_action, _ = self._act(env)   # The next skill acts on this step already
        elif self.step_index < len(self.plan):
            self.skill_action, done = self._act(env)
            if done:
                self._next(env)
                if self._skill() == "dance":
                    return self._dance_step(env)
                if self.step_index < len(self.plan):
                    self.skill_action, _ = self._act(env)
        if self.step_index < len(self.plan):
            lo, hi = DEPTH_LIMITS[self.plan[self.step_index][0]]
            action = np.append(np.clip(self.skill_action, -1.0, 1.0), 0.0)   # No sway
            action[1] = float(np.clip(action[1], self._depth_action(lo), self._depth_action(hi)))
        else:
            action = self._hold_action(env, self.hold_heading)
        self.last_action = action
        return env.step(action)
