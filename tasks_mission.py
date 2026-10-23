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
"""
import random
from typing import List, Optional

from sub_env import SubTaskEnv

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
