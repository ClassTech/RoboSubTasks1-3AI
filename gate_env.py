#!/usr/bin/env python3
"""
Gymnasium environment for the gate skill: start somewhere in front of the gate
and go through it. Actions, observation and the episode loop are in sub_env.py.

Going forward through the gate opening pays +20. The episode succeeds (+5)
once the sub carries on to CLEAR_M beyond the gate, clear of its frame, so the
policy learns to keep going straight instead of turning back. Failures (-5
each): hitting something, crossing the gate line anywhere else or crossing
back, or running out of time.
"""
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import DEPTH_MAX, DEPTH_MIN, HEADING_CHANGE_DEG, SubTaskEnv  # noqa: F401  (re-exported)

CLEAR_M = 1.5       # Beyond the gate line, past the frame, before the skill counts as done


@dataclass
class StartConfig:
    """Where episodes start, relative to the gate. Each value is drawn uniformly from (min, max)."""
    distance_m: tuple = (3.0, 8.0)        # In front of the gate
    lateral_m: tuple = (-1.5, 1.5)        # Sideways from the gate's center line
    heading_deg: tuple = (-30.0, 30.0)    # Relative to facing the gate squarely
    depth_m: tuple = (0.1, 0.5)
    # Search starts (training only): a share of episodes turned away so the gate starts out of view
    search_share: float = 0.0
    search_turn_deg: tuple = (40.0, 110.0)   # Extra heading, either way


@dataclass
class RewardConfig:
    progress: float = 1.0          # Per meter closer to the gate center (horizontal + depth + center line)
    depth_weight: float = 1.0      # How much depth error counts in that distance
    center_line_weight: float = 0.5  # How much distance from the gate's center line counts
    target_depth_m: float = 1.5    # Middle of the usable gate opening
    gate_bonus: float = 20.0       # For the pass, paid through the progress measure
    clear_bonus: float = 5.0       # For carrying on to CLEAR_M beyond the gate
    collision_penalty: float = 5.0
    missed_gate_penalty: float = 5.0
    timeout_penalty: float = 5.0   # So hovering near the gate isn't a safe alternative to committing
    time_penalty: float = 0.01     # Per step
    energy_penalty: float = 0.05   # Per second at full thrust on every thruster


class GateEnv(SubTaskEnv):
    OUTCOMES = ("is_success", "collision", "missed_gate", "surfaced", "timeout")
    MIN_TARGET_DEPTH_M = 1.2   # The opening is below 1.15 m, so the gate never needs to aim shallower
    MAX_TARGET_DEPTH_M = 1.65  # At least 0.44 m above the pole feet (floor 2.09 m), so perception can range them.
                               # Allowed down to 1.9 m, a policy got stuck there (its depth output clipped at
                               # the limit gives no learning signal) and misjudged the gate by 2-3 m

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 60.0, start: Optional[StartConfig] = None,
                 reward: Optional[RewardConfig] = None, randomize: bool = False,
                 mission_return_starts: Optional[list] = None, mission_share: float = 0.3):
        self.start_cfg = start or StartConfig()
        # Recorded handoffs into the return leg (training); those episodes succeed going back through
        self.mission_return_starts, self.mission_share = mission_return_starts or [], mission_share
        self.expect_direction = 1
        self.reward_cfg = rc = reward or RewardConfig()
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=rc.progress,
                         timeout_penalty=rc.timeout_penalty, randomize=randomize)

    def _place_sub(self, rng: random.Random):
        s, g, sc = self.core.state, self.core.gate, self.start_cfg
        self.expect_direction = 1
        if self.mission_return_starts and rng.random() < self.mission_share:
            self.apply_recorded_start(rng.choice(self.mission_return_starts))
            self.expect_direction = -1
        else:
            s.x = g.x - rng.uniform(*sc.distance_m)
            s.y = g.center_y + rng.uniform(*sc.lateral_m)
            s.heading = rng.uniform(*sc.heading_deg) % 360
            s.z = rng.uniform(*sc.depth_m)
        if sc.search_share and rng.random() < sc.search_share:
            # Turned away: the policy has to find the gate before it can line up
            s.heading = (s.heading + rng.choice((-1, 1)) * rng.uniform(*sc.search_turn_deg)) % 360

    def _on_reset(self):
        self.passed, self._at_pass = False, None

    def _past_gate_m(self) -> float:
        """How far beyond the gate line the sub is, in the direction it should pass. Ground truth."""
        return (self.core.state.x - self.core.gate.x) * self.expect_direction

    def _physics_outcome(self) -> Optional[str]:
        for c in self.new_crossings():
            if not self.passed and c.inside_opening and c.direction == self.expect_direction:
                self.passed = True
            else:
                return "missed_gate"
        return None

    def _complete(self) -> bool:
        return self.passed and self._past_gate_m() >= CLEAR_M

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return {"success": rc.clear_bonus, "collision": -rc.collision_penalty,
                "missed_gate": -rc.missed_gate_penalty}[outcome]

    def _potential(self) -> float:
        """Before the pass: minus the distance to the gate center (horizontal, plus weighted depth error
        and distance from the gate's center line, so approaching square-on pays). After: its value at the
        pass, plus the gate bonus, plus progress out to CLEAR_M beyond. Ground truth."""
        rc = self.reward_cfg
        if self.passed:
            if self._at_pass is None:
                self._at_pass = self._approach_potential()
            return self._at_pass + rc.gate_bonus + min(max(self._past_gate_m(), 0.0), CLEAR_M)
        return self._approach_potential()

    def _approach_potential(self) -> float:
        s, g, rc = self.core.state, self.core.gate, self.reward_cfg
        horizontal = float(np.hypot(g.x - s.x, g.center_y - s.y))
        return -(horizontal + rc.depth_weight * abs(s.z - rc.target_depth_m)
                 + rc.center_line_weight * abs(s.y - g.center_y))

    def scripted_mission(self) -> List:
        from ai.tasks import GateTask
        return [GateTask(target_depth=1.5)]


@dataclass
class ReturnStartConfig:
    """Where the orbit hands off (measured: ~10 m past the gate, 1.2-3.1 m to one side,
    within ±24° of facing it, 1.1-1.9 m deep), a little wider. Distance and lateral are
    measured from the gate on the far side; heading relative to facing back through it."""
    distance_m: tuple = (8.0, 11.0)
    lateral_m: tuple = (-3.5, 3.5)
    heading_deg: tuple = (-30.0, 30.0)
    depth_m: tuple = (1.1, 1.9)


class GateReturnEnv(GateEnv):
    """The gate skill on the way back: start where the orbit hands off and go back through."""

    def __init__(self, detector_mode: str = "perfect", start: Optional[ReturnStartConfig] = None, **kwargs):
        super().__init__(detector_mode, **kwargs)
        self.start_cfg = start or ReturnStartConfig()

    def _place_sub(self, rng: random.Random):
        s, g, sc = self.core.state, self.core.gate, self.start_cfg
        self.expect_direction = -1
        s.x = g.x + rng.uniform(*sc.distance_m)
        s.y = g.center_y + rng.uniform(*sc.lateral_m)
        s.heading = (180.0 + rng.uniform(*sc.heading_deg)) % 360
        s.z = rng.uniform(*sc.depth_m)

